//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_dep.cpp
//! \brief Additive ghost-zone ("deposit") exchange for cell-centered fields built by
//! particle-mesh deposition. A particle near a MeshBlock boundary deposits part of its
//! weight into ghost cells of its own MeshBlock; those contributions physically belong
//! to the *active* cells of the neighboring MeshBlock and must be added there. Thus the
//! data flow is the reverse of the ordinary copy exchange: the sender packs its GHOST
//! regions and the receiver *sums* the buffers into its active-zone edge strips.
//!
//! Because the receive strips of different neighbor buffers overlap (a face strip and
//! an edge strip share cells near active-zone corners), the receiver must accumulate
//! buffers with a scalar loop over neighbors, exactly as in
//! MeshBoundaryValuesFC::SumBoundaryFluxes().
//!
//! Static mesh refinement (level jumps of one between neighbors): every deposited field
//! is a density (mass, momentum or a rate per unit volume), and every block deposits
//! with the kernel of the FINEST level (dust::DepositStencil: a block one level below the
//! finest scatters into a 2x fine image of itself, which is then volume-averaged onto its
//! cells).  The exchange across a level boundary is then both conservative and
//! consistent:
//!   * a FINE block sends its ghost shell RESTRICTED to the coarse resolution (the
//!     volume average of the 2^d fine ghost cells covering each coarse cell: the mass
//!     m W of the fine cells is summed and divided by the coarse volume);
//!   * a COARSE block sends the 2^d fine-image cells of each of its ghost cells (the
//!     finest-kernel deposits of its own particles), which the fine receiver adds into
//!     the corresponding fine cells.
//! A fine cell thus receives the finest-kernel weight of every nearby particle and a
//! coarse cell the volume average of those weights, whatever block the particles live
//! in; a uniform particle lattice at the finest spacing deposits an exactly uniform
//! field across the interface.
//! The index sets follow the neighbor table of MeshBlock::SetNeighbors: the buffer of a
//! coarser neighbor is the subblock slot (f1,f2) of the fine block's position inside its
//! parent, and a fine block's edge/corner slot toward a coarser neighbor is only set at
//! the EXTERIOR edges/corners of the coarse face -- an interior edge belongs to the same
//! coarse block as the face, so the face buffer carries the fine ghost shell's
//! transverse overhang on the interior side (ng fine = ng/2 coarse cells), exactly the
//! adjoint of the copy exchange's (cnx - ng) overhang.  The receiving coarse block sums
//! it into the cng cells just across the midline of its face.
//!
//! Shear-periodic x1 faces (3D shearing box): the x1 ghost slabs of the face blocks are
//! NOT folded by the plain-periodic pass (their unsheared x1 buffers are skipped in
//! RecvAndSumDeposit); FoldShearDeposit gathers them into global y-z planes, remaps
//! each row in y by -/+yshear with the conservative remap-flux kernels of the orbital
//! advection / multigrid shear machinery (shearing_box/remap_fluxes.hpp), and adds the
//! result into the active edge strips of the blocks across the face.  Sign: the copy
//! exchange fills inner ghosts with outer content shifted by +yshear; the deposit map
//! is its adjoint, so inner-ghost deposits are shifted by -yshear (outer by +yshear).
//! With refinement the face blocks share one level (Mesh::CheckShearingBoxRefinement)
//! and the planes are laid out at that level.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "shearing_box/remap_fluxes.hpp"
#include "bvals.hpp"

namespace {
// halo of the y-remap scratch rows: PPMX needs j-2..j+2 plus the face at ju+1 (the
// integer part of the shift is folded into the periodic-wrapped scratch load)
constexpr int PAD = 3;
}

//----------------------------------------------------------------------------------------
// MeshBoundaryValuesDep constructor:

MeshBoundaryValuesDep::MeshBoundaryValuesDep(MeshBlockPack *pp, ParameterInput *pin) :
  MeshBoundaryValues(pp, pin, false),
  coarse_("dep_coarse",1,1,1,1,1) {
  Mesh *pm = pp->pmesh;
  multilevel_ = pm->multilevel;
  nsub_ = pm->three_d ? 8 : (pm->multi_d ? 4 : 2);
  if (multilevel_) {
    // the injection of a coarser neighbor's ng ghost cells needs ng coarse cells (2ng
    // fine cells) inside the fine block's active zone
    auto &mb = pm->mb_indcs;
    if ((2*mb.ng > mb.nx1) || (pm->multi_d && (2*mb.ng > mb.nx2)) ||
        (pm->three_d && (2*mb.ng > mb.nx3))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Additive deposit exchange with SMR needs MeshBlocks of "
                << "at least 2*nghost cells in each active dimension" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // shear-periodic x1 faces: global plane geometry, per-block global offsets, and the
  // x1-direction table of the buffer indices (inverse of NeighborIndex)
  shear_x1_ = (pm->three_d &&
               (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic));
  if (shear_x1_) {
    // level of the shear-face MeshBlocks (uniform along both faces by the mesh policy)
    shear_lev_ = pm->root_level;
    for (int mm=0; mm<(pm->nmb_total); ++mm) {
      if (pm->lloc_eachmb[mm].lx1 == 0) {shear_lev_ = pm->lloc_eachmb[mm].level; break;}
    }
    int lshift = shear_lev_ - pm->root_level;
    shear_gny_ = (pm->mesh_indcs.nx2) << lshift;   // global cell counts at shear_lev_
    shear_gnz_ = (pm->mesh_indcs.nx3) << lshift;
    x3_periodic_ = (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);
    InitShearOffsets();

    // x1 direction of each buffer index, from the layout documented in
    // mesh/nghbr_index.hpp: x1 faces [0-3] (-) [4-7] (+); x2 faces [8-15] (0);
    // x1x2 edges [16-23] = 16 + (ix+1) + 2*(iy+1) + n1; x3 faces [24-31] (0);
    // x3x1 edges [32-39] = 24 + |ix|*(ix+9) + 2*(iz+1) + n1; x2x3 edges [40-47] (0);
    // corners [48-55] = 48 + (ix+1)/2 + (iy+1) + 2*(iz+1)
    Kokkos::realloc(nghbr_ox1_, 56);
    auto ox1_h = Kokkos::create_mirror_view(nghbr_ox1_);
    for (int n=0; n<56; ++n) {
      int ox1 = 0;
      if (n < 4) {
        ox1 = -1;
      } else if (n < 8) {
        ox1 = +1;
      } else if ((n >= 16) && (n < 24)) {
        ox1 = (((n - 16) % 4) >= 2) ? +1 : -1;
      } else if ((n >= 32) && (n < 40)) {
        ox1 = (((n - 32) % 4) >= 2) ? +1 : -1;
      } else if (n >= 48) {
        ox1 = (((n - 48) % 2) == 1) ? +1 : -1;
      }
      ox1_h(n) = ox1;
    }
    // cross-check the face and corner entries against NeighborIndex itself
    if ((ox1_h(NeighborIndex(-1,0,0,0,0)) != -1) || (ox1_h(NeighborIndex(1,0,0,0,0)) != 1)
        || (ox1_h(NeighborIndex(-1,-1,-1,0,0)) != -1)
        || (ox1_h(NeighborIndex(1,1,1,0,0)) != 1)
        || (ox1_h(NeighborIndex(-1,0,1,0,0)) != -1)
        || (ox1_h(NeighborIndex(1,-1,0,1,0)) != 1)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "neighbor-buffer x1-direction table inconsistent with "
                << "NeighborIndex" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Kokkos::deep_copy(nghbr_ox1_, ox1_h);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesDep::InitShearOffsets()
//! \brief Per-block global (x2, x3) cell offsets of this pack's MeshBlocks in the shear
//! planes (used for the face blocks, which sit at shear_lev_).  Rebuilt after AMR.

void MeshBoundaryValuesDep::InitShearOffsets() {
  if (!shear_x1_) return;
  MeshBlockPack *pp = pmy_pack;
  Mesh *pm = pp->pmesh;
  int nmb = pp->nmb_thispack;
  Kokkos::realloc(shear_goffs_, std::max(nmb, 1), 2);
  auto goffs_h = Kokkos::create_mirror_view(shear_goffs_);
  for (int m=0; m<nmb; ++m) {
    LogicalLocation &lloc = pm->lloc_eachmb[m + pp->gids];
    goffs_h(m,0) = static_cast<int>(lloc.lx2)*pm->mb_indcs.nx2;
    goffs_h(m,1) = static_cast<int>(lloc.lx3)*pm->mb_indcs.nx3;
  }
  Kokkos::deep_copy(shear_goffs_, goffs_h);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesDep::InitSendIndices
//! \brief Calculates indices of GHOST cells packed into send buffers. These are the
//! mirror image of the receive ("unpack into ghosts") indices of the copy exchange in
//! MeshBoundaryValuesCC::InitRecvIndices.
//!   isame: same-level neighbor, own resolution, the ghost slab/edge/corner
//!   icoar: COARSER neighbor: the ghost shell in the block's COARSE index space
//!          (packed from the restricted copy), ng/2 coarse cells deep, plus the
//!          transverse overhang of ng/2 coarse cells on the interior side(s) of the
//!          parent (f = 0 -> the block is the lower child -> overhang upward)
//!   ifine: FINER neighbor: own ghost cells, ng deep, over the transverse half (f1,f2)
//!          covered by that fine neighbor, each sent as its 2^d fine-image cells

void MeshBoundaryValuesDep::InitSendIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb = pmy_pack->pmesh->mb_indcs;
  int ng = mb.ng;
  int cng = ng/2;

  // same level (slot (0,0) only)
  if ((f1 == 0) && (f2 == 0)) {
    auto &isame = buf.isame[0];
    if (ox1 == 0) {
      isame.bis = mb.is;          isame.bie = mb.ie;
    } else if (ox1 > 0) {
      isame.bis = mb.ie + 1;      isame.bie = mb.ie + ng;
    } else {
      isame.bis = mb.is - ng;     isame.bie = mb.is - 1;
    }
    if (ox2 == 0) {
      isame.bjs = mb.js;          isame.bje = mb.je;
    } else if (ox2 > 0) {
      isame.bjs = mb.je + 1;      isame.bje = mb.je + ng;
    } else {
      isame.bjs = mb.js - ng;     isame.bje = mb.js - 1;
    }
    if (ox3 == 0) {
      isame.bks = mb.ks;          isame.bke = mb.ke;
    } else if (ox3 > 0) {
      isame.bks = mb.ke + 1;      isame.bke = mb.ke + ng;
    } else {
      isame.bks = mb.ks - ng;     isame.bke = mb.ks - 1;
    }
    buf.isame_ndat = (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*
                     (isame.bke - isame.bks + 1);
  }
  if (!multilevel_) {return;}

  // the transverse subblock index of each direction (as in MeshBoundaryValuesCC):
  // x <- f1 (when ox1 == 0); y <- f1 if ox1 != 0 else f2; z <- f1 on x1x2 edges else f2
  const int fx = f1;
  const int fy = (ox1 != 0) ? f1 : f2;
  const int fz = (ox1 != 0 && ox2 != 0) ? f1 : f2;

  // to a COARSER neighbor: restricted ghost shell (coarse indices) + interior overhang
  {auto &ic = buf.icoar[0];
  if (ox1 == 0) {
    ic.bis = mb.cis;              ic.bie = mb.cie;
    if (fx == 0) {ic.bie += cng;} else {ic.bis -= cng;}
  } else if (ox1 > 0) {
    ic.bis = mb.cie + 1;          ic.bie = mb.cie + cng;
  } else {
    ic.bis = mb.cis - cng;        ic.bie = mb.cis - 1;
  }
  if (ox2 == 0) {
    ic.bjs = mb.cjs;              ic.bje = mb.cje;
    if (mb.nx2 > 1) {if (fy == 0) {ic.bje += cng;} else {ic.bjs -= cng;}}
  } else if (ox2 > 0) {
    ic.bjs = mb.cje + 1;          ic.bje = mb.cje + cng;
  } else {
    ic.bjs = mb.cjs - cng;        ic.bje = mb.cjs - 1;
  }
  if (ox3 == 0) {
    ic.bks = mb.cks;              ic.bke = mb.cke;
    if (mb.nx3 > 1) {if (fz == 0) {ic.bke += cng;} else {ic.bks -= cng;}}
  } else if (ox3 > 0) {
    ic.bks = mb.cke + 1;          ic.bke = mb.cke + cng;
  } else {
    ic.bks = mb.cks - cng;        ic.bke = mb.cks - 1;
  }
  buf.icoar_ndat = (ic.bie - ic.bis + 1)*(ic.bje - ic.bjs + 1)*(ic.bke - ic.bks + 1);
  }

  // to a FINER neighbor: own ghost cells over the transverse half of that neighbor
  {auto &fi = buf.ifine[0];
  if (ox1 == 0) {
    fi.bis = mb.is;               fi.bie = mb.ie;
    if (fx == 1) {fi.bis += mb.cnx1;} else {fi.bie -= mb.cnx1;}
  } else if (ox1 > 0) {
    fi.bis = mb.ie + 1;           fi.bie = mb.ie + ng;
  } else {
    fi.bis = mb.is - ng;          fi.bie = mb.is - 1;
  }
  if (ox2 == 0) {
    fi.bjs = mb.js;               fi.bje = mb.je;
    if (mb.nx2 > 1) {if (fy == 1) {fi.bjs += mb.cnx2;} else {fi.bje -= mb.cnx2;}}
  } else if (ox2 > 0) {
    fi.bjs = mb.je + 1;           fi.bje = mb.je + ng;
  } else {
    fi.bjs = mb.js - ng;          fi.bje = mb.js - 1;
  }
  if (ox3 == 0) {
    fi.bks = mb.ks;               fi.bke = mb.ke;
    if (mb.nx3 > 1) {if (fz == 1) {fi.bks += mb.cnx3;} else {fi.bke -= mb.cnx3;}}
  } else if (ox3 > 0) {
    fi.bks = mb.ke + 1;           fi.bke = mb.ke + ng;
  } else {
    fi.bks = mb.ks - ng;          fi.bke = mb.ks - 1;
  }
  // each ghost cell is sent as its nsub_ fine-image cells
  buf.ifine_ndat = nsub_*(fi.bie - fi.bis + 1)*(fi.bje - fi.bjs + 1)*
                   (fi.bke - fi.bks + 1);
  }
  // Payload sizes: icoar (sender finer) matches the receiver's ifine, and ifine (sender
  // coarser) matches the receiver's icoar, by construction of InitRecvIndices().
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesDep::InitRecvIndices
//! \brief Calculates indices of ACTIVE cells into which receive buffers are summed.
//!   isame: same-level neighbor: the edge strip ng deep
//!   ifine: FINER neighbor: own active cells ng/2 deep over the transverse half (f1,f2)
//!          of that neighbor, extended by ng/2 cells across the midline (the fine
//!          block's interior overhang)
//!   icoar: COARSER neighbor: indices are in this block's COARSE index space (ng coarse
//!          cells deep, whole transverse range); the buffer holds the 2^d fine-image
//!          cells of each such coarse cell, added into the fine cells they cover

void MeshBoundaryValuesDep::InitRecvIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb = pmy_pack->pmesh->mb_indcs;
  int ng = mb.ng;
  int ng1 = ng - 1;
  int cng = ng/2;

  if ((f1 == 0) && (f2 == 0)) {
    auto &isame = buf.isame[0];
    isame.bis = (ox1 > 0) ? (mb.ie - ng1) : mb.is;
    isame.bie = (ox1 < 0) ? (mb.is + ng1) : mb.ie;
    isame.bjs = (ox2 > 0) ? (mb.je - ng1) : mb.js;
    isame.bje = (ox2 < 0) ? (mb.js + ng1) : mb.je;
    isame.bks = (ox3 > 0) ? (mb.ke - ng1) : mb.ks;
    isame.bke = (ox3 < 0) ? (mb.ks + ng1) : mb.ke;
    buf.isame_ndat = (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*
                     (isame.bke - isame.bks + 1);
  }
  if (!multilevel_) {return;}

  const int fx = f1;
  const int fy = (ox1 != 0) ? f1 : f2;
  const int fz = (ox1 != 0 && ox2 != 0) ? f1 : f2;

  // from a FINER neighbor: own active strip cng deep, its half of the face + overhang
  {auto &fi = buf.ifine[0];
  if (ox1 == 0) {
    fi.bis = mb.is;               fi.bie = mb.ie;
    if (fx == 0) {fi.bie -= (mb.cnx1 - cng);} else {fi.bis += (mb.cnx1 - cng);}
  } else if (ox1 > 0) {
    fi.bis = mb.ie - cng + 1;     fi.bie = mb.ie;
  } else {
    fi.bis = mb.is;               fi.bie = mb.is + cng - 1;
  }
  if (ox2 == 0) {
    fi.bjs = mb.js;               fi.bje = mb.je;
    if (mb.nx2 > 1) {
      if (fy == 0) {fi.bje -= (mb.cnx2 - cng);} else {fi.bjs += (mb.cnx2 - cng);}
    }
  } else if (ox2 > 0) {
    fi.bjs = mb.je - cng + 1;     fi.bje = mb.je;
  } else {
    fi.bjs = mb.js;               fi.bje = mb.js + cng - 1;
  }
  if (ox3 == 0) {
    fi.bks = mb.ks;               fi.bke = mb.ke;
    if (mb.nx3 > 1) {
      if (fz == 0) {fi.bke -= (mb.cnx3 - cng);} else {fi.bks += (mb.cnx3 - cng);}
    }
  } else if (ox3 > 0) {
    fi.bks = mb.ke - cng + 1;     fi.bke = mb.ke;
  } else {
    fi.bks = mb.ks;               fi.bke = mb.ks + cng - 1;
  }
  buf.ifine_ndat = (fi.bie - fi.bis + 1)*(fi.bje - fi.bjs + 1)*(fi.bke - fi.bks + 1);
  }

  // from a COARSER neighbor: coarse cells to inject, ng deep at the edge (coarse indices)
  {auto &ic = buf.icoar[0];
  if (ox1 == 0) {
    ic.bis = mb.cis;              ic.bie = mb.cie;
  } else if (ox1 > 0) {
    ic.bis = mb.cie - ng + 1;     ic.bie = mb.cie;
  } else {
    ic.bis = mb.cis;              ic.bie = mb.cis + ng - 1;
  }
  if (ox2 == 0) {
    ic.bjs = mb.cjs;              ic.bje = mb.cje;
  } else if (ox2 > 0) {
    ic.bjs = mb.cje - ng + 1;     ic.bje = mb.cje;
  } else {
    ic.bjs = mb.cjs;              ic.bje = mb.cjs + ng - 1;
  }
  if (ox3 == 0) {
    ic.bks = mb.cks;              ic.bke = mb.cke;
  } else if (ox3 > 0) {
    ic.bks = mb.cke - ng + 1;     ic.bke = mb.cke;
  } else {
    ic.bks = mb.cks;              ic.bke = mb.cks + ng - 1;
  }
  // each coarse cell of the range arrives as its nsub_ fine cells
  buf.icoar_ndat = nsub_*(ic.bie - ic.bis + 1)*(ic.bje - ic.bjs + 1)*
                   (ic.bke - ic.bks + 1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesDep::RestrictDeposit
//! \brief Volume-average the deposit array into the internal coarse copy over the coarse
//! active zone AND the coarse ghost shell (ng/2 cells deep, the image of the ng fine
//! ghost cells): sends to coarser neighbors are packed from the ghost part.

void MeshBoundaryValuesDep::RestrictDeposit(DvceArray5D<Real> &a) {
  auto &mb = pmy_pack->pmesh->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int nvar = a.extent_int(1);
  const int ng = mb.ng, cng = ng/2;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  if (coarse_nvar_ != nvar) {
    int nmbmax = std::max(nmb, pmy_pack->pmesh->nmb_maxperrank);
    int n1 = mb.cnx1 + 2*ng;
    int n2 = (mb.nx2 > 1) ? (mb.cnx2 + 2*ng) : 1;
    int n3 = (mb.nx3 > 1) ? (mb.cnx3 + 2*ng) : 1;
    Kokkos::realloc(coarse_, nmbmax, nvar, n3, n2, n1);
    coarse_nvar_ = nvar;
  }
  const int cis = mb.cis, cie = mb.cie, cjs = mb.cjs, cje = mb.cje;
  const int cks = mb.cks, cke = mb.cke;
  const int is = mb.is, js = mb.js, ks = mb.ks;
  const int il = cis - cng, iu = cie + cng;
  const int jl = multi_d ? (cjs - cng) : cjs, ju = multi_d ? (cje + cng) : cje;
  const int kl = three_d ? (cks - cng) : cks, ku = three_d ? (cke + cng) : cke;
  auto ca = coarse_;
  par_for("dep_restrict", DevExeSpace(), 0, nmb-1, 0, nvar-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    const int fi = 2*(i - cis) + is;
    if (three_d) {
      const int fj = 2*(j - cjs) + js;
      const int fk = 2*(k - cks) + ks;
      ca(m,v,k,j,i) = 0.125*(a(m,v,fk  ,fj  ,fi) + a(m,v,fk  ,fj  ,fi+1)
                           + a(m,v,fk  ,fj+1,fi) + a(m,v,fk  ,fj+1,fi+1)
                           + a(m,v,fk+1,fj  ,fi) + a(m,v,fk+1,fj  ,fi+1)
                           + a(m,v,fk+1,fj+1,fi) + a(m,v,fk+1,fj+1,fi+1));
    } else if (multi_d) {
      const int fj = 2*(j - cjs) + js;
      ca(m,v,k,j,i) = 0.25*(a(m,v,k,fj  ,fi) + a(m,v,k,fj  ,fi+1)
                          + a(m,v,k,fj+1,fi) + a(m,v,k,fj+1,fi+1));
    } else {
      ca(m,v,k,j,i) = 0.5*(a(m,v,k,j,fi) + a(m,v,k,j,fi+1));
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MeshBoundaryValuesDep::PackAndSendDeposit()
//! \brief Pack ghost-region deposits into boundary buffers and send to neighbors.
//! Adapted from MeshBoundaryValuesCC::PackAndSendCC.  Same-rank neighbors are written
//! directly into the destination receive buffer; the task graph guarantees the receiver
//! does not start summing until all local packs are complete.  Sends to coarser
//! neighbors are packed from the restricted copy (RestrictDeposit).
//!
//! Input array must be a 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)

TaskStatus MeshBoundaryValuesDep::PackAndSendDeposit(DvceArray5D<Real> &a,
                                                     DvceArray5D<Real> &fimg) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);
  if (multilevel_) {RestrictDeposit(a);}

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
  auto ca = coarse_;
  const int nsub = nsub_;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const bool have_rfac = have_rfac_;
  auto rf = rfac_.d_view;
#if MPI_PARALLEL_ENABLED
  // Build (or refresh) the rank-packed metadata before the kernel writes off-rank
  // payloads directly into the aggregate send buffer.
  if (rank_packed_bvals_nvars_ != nvar ||
      rank_packed_mesh_seq_ != pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq()) {
    BuildRankPackedVarMetadata(nvar);
  }
  auto aggsbuf = rank_sendbuf_vars_;
  auto sendoff = send_agg_offset_;
#endif
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*nnghbr*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("DepSendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      // neighbor coarser: pack the restricted ghost shell; same: own ghosts; finer:
      // the fine-image cells of own ghosts over the neighbor's half of the face
      const bool to_coarse = (nghbr.d_view(m,n).lev < mblev.d_view(m));
      const bool to_fine = (nghbr.d_view(m,n).lev > mblev.d_view(m));
      const MeshBufferIndcs &ix = to_coarse ? sbuf[n].icoar[0] :
          (to_fine ? sbuf[n].ifine[0] : sbuf[n].isame[0]);
      int il = ix.bis, iu = ix.bie;
      int jl = ix.bjs, ju = ix.bje;
      int kl = ix.bks, ku = ix.bke;
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj  = nk*nj;
      const int ns = to_fine ? nsub : 1;   // values per (coarse) cell of the range

      // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
      // in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m,n).dest;

      // value of cell (k,j,i) of the range; when to_fine, sub-cell q of the cell at the
      // NEIGHBOUR's level (2^d per cell) = mean of the (r/2)^d image cells it covers,
      // r = this block's image factor (2 with one level: the image cells themselves)
      const int r = have_rfac ? rf(m) : 2;
      const int h = r/2;
      auto value = [&](const int k, const int j, const int i, const int q) -> Real {
        if (to_coarse) {return ca(m,v,k,j,i);}
        if (!to_fine) {return a(m,v,k,j,i);}
        const int si = q & 1;
        const int sj = multi_d ? ((q >> 1) & 1) : 0;
        const int sk = three_d ? ((q >> 2) & 1) : 0;
        const int i0 = r*i + si*h;
        const int j0 = multi_d ? r*j + sj*h : j;
        const int k0 = three_d ? r*k + sk*h : k;
        const int hj = multi_d ? h : 1, hk = three_d ? h : 1;
        Real sum = 0.0;
        for (int kk=0; kk<hk; ++kk) {
          for (int jj=0; jj<hj; ++jj) {
            for (int ii=0; ii<h; ++ii) {
              sum += fimg(m, v, k0 + kk, j0 + jj, i0 + ii);
            }
          }
        }
        return sum/static_cast<Real>(h*hj*hk);
      };

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i (and the fine-image sub-cells)
        // copy directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il*ns,(iu+1)*ns),
          [&](const int iq) {
            const int i = iq/ns, q = iq - i*ns;
            rbuf[dn].vars(dm, ns*(i-il + ni*(j-jl + nj*(k-kl + nk*v))) + q) =
                value(k,j,i,q);
          });
        // else copy into send buffer for MPI communication below
        } else {
#if MPI_PARALLEL_ENABLED
          int base = sendoff(m*nnghbr + n);
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il*ns,(iu+1)*ns),
          [&](const int iq) {
            const int i = iq/ns, q = iq - i*ns;
            aggsbuf(base + ns*(i-il + ni*(j-jl + nj*(k-kl + nk*v))) + q) =
                value(k,j,i,q);
          });
#else
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il*ns,(iu+1)*ns),
          [&](const int iq) {
            const int i = iq/ns, q = iq - i*ns;
            sbuf[n].vars(m, ns*(i-il + ni*(j-jl + nj*(k-kl + nk*v))) + q) =
                value(k,j,i,q);
          });
#endif
        }
      });
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer
  }

#if MPI_PARALLEL_ENABLED
  // Send one aggregate boundary payload to each neighboring MPI rank. The packing
  // kernel wrote off-rank entries directly into rank_sendbuf_vars_.
  Kokkos::fence();
  bool no_errors = true;
  std::fill(send_var_reqs_.begin(), send_var_reqs_.end(), MPI_REQUEST_NULL);
  for (std::size_t i = 0; i < send_var_msgs_.size(); ++i) {
    const auto &msg = send_var_msgs_[i];
    int ierr = MPI_Isend(rank_sendbuf_vars_.data() + msg.offset, msg.data_size,
                         MPI_ATHENA_REAL, msg.rank, 1, comm_vars, &send_var_reqs_[i]);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MeshBoundaryValuesDep::RecvAndSumDeposit()
//! \brief Check receives are complete, then SUM buffers into active-zone edge strips.
//! The loop over neighbor buffers is intentionally scalar (sequential within each team):
//! receive strips of different buffers overlap near active-zone edges/corners, so
//! concurrent accumulation over buffers would race (cf. SumBoundaryFluxes).  Buffers
//! from a coarser neighbor hold coarse cells: each is injected (added) into the 2^d
//! fine cells it covers.

TaskStatus MeshBoundaryValuesDep::RecvAndSumDeposit(DvceArray5D<Real> &a) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (std::size_t i = 0; i < recv_var_reqs_.size(); ++i) {
    int test;
    int ierr = MPI_Test(&recv_var_reqs_[i], &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so sum into active cells

  int nvar = a.extent_int(1);
#if MPI_PARALLEL_ENABLED
  auto aggrbuf = rank_recvbuf_vars_;
  auto recvoff = recv_agg_offset_;
#endif
  // shear-periodic x1: the buffers that cross a shear face of this block carry the
  // UNSHEARED periodic image (the tree wraps shear_periodic like periodic); their
  // contribution is delivered by FoldShearDeposit instead, so skip them here
  const bool shear = shear_x1_;
  auto ox1tab = nghbr_ox1_;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  // fine-index mapping of the injection from coarser neighbors
  auto &mb = pmy_pack->pmesh->mb_indcs;
  const int cis = mb.cis, cjs = mb.cjs, cks = mb.cks;
  const int is = mb.is, js = mb.js, ks = mb.ks;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int nsub = nsub_;

  // Outer loop over (# of MeshBlocks)*(# of variables); loop over buffers is scalar
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("DepRecvSum", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/nvar;
    const int v = (tmember.league_rank() - m*nvar);

    for (int n=0; n<nnghbr; ++n) {
      bool skip = false;
      if (shear) {
        int ox1 = ox1tab(n);
        skip = ((ox1 < 0) && (mb_bcs.d_view(m,BoundaryFace::inner_x1) ==
                              BoundaryFlag::shear_periodic)) ||
               ((ox1 > 0) && (mb_bcs.d_view(m,BoundaryFace::outer_x1) ==
                              BoundaryFlag::shear_periodic));
      }
      // only sum buffers when neighbor exists (and not across a shear face)
      if ((nghbr.d_view(m,n).gid >= 0) && !skip) {
        const bool from_coarse = (nghbr.d_view(m,n).lev < mblev.d_view(m));
        const MeshBufferIndcs &ix = from_coarse ? rbuf[n].icoar[0] :
            ((nghbr.d_view(m,n).lev == mblev.d_view(m)) ? rbuf[n].isame[0] :
                                                           rbuf[n].ifine[0]);
        int il = ix.bis, iu = ix.bie;
        int jl = ix.bjs, ju = ix.bje;
        int kl = ix.bks, ku = ix.bke;
        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj  = nk*nj;
        const int ns = from_coarse ? nsub : 1;   // values per (coarse) cell of the range
#if MPI_PARALLEL_ENABLED
        // A non-negative aggregate offset identifies an off-rank payload. Same-rank
        // payloads remain in their per-neighbor receive buffers.
        const int base = recvoff(m*nnghbr + n);
#endif

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // Inner (vector) loop over i (and the fine sub-cells of a coarse-range cell)
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il*ns,(iu+1)*ns),
          [&](const int iq) {
            const int i = iq/ns, q = iq - i*ns;
            const int bi = ns*(i-il + ni*(j-jl + nj*(k-kl + nk*v))) + q;
#if MPI_PARALLEL_ENABLED
            const Real val = (base >= 0) ? aggrbuf(base + bi) : rbuf[n].vars(m, bi);
#else
            const Real val = rbuf[n].vars(m, bi);
#endif
            if (!from_coarse) {
              a(m,v,k,j,i) += val;
            } else {
              // (k,j,i) are COARSE indices of this block and q the fine sub-cell: add
              // the sender's fine-image value into that fine cell
              const int si = q & 1;
              const int sj = multi_d ? ((q >> 1) & 1) : 0;
              const int sk = three_d ? ((q >> 2) & 1) : 0;
              const int fi = 2*(i - cis) + is + si;
              const int fj = multi_d ? (2*(j - cjs) + js + sj) : j;
              const int fk = three_d ? (2*(k - cks) + ks + sk) : k;
              a(m,v,fk,fj,fi) += val;
            }
          });
        });
      }
      // barrier so no thread starts summing the next (overlapping) buffer early
      tmember.team_barrier();
    }
  });  // end par_for_outer

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MeshBoundaryValuesDep::FoldShearDeposit()
//! \brief Fold the x1 ghost slabs of the shear-face MeshBlocks across the shear-periodic
//! x1 faces: gather (additively) into two global y-z planes, remap every plane row in y
//! by -yshear (inner-ghost deposits -> outer active cells) resp. +yshear (outer-ghost
//! deposits -> inner active cells), and ADD the remapped planes into the active edge
//! strips.  The whole slab (k, j incl. the y/z corner ghosts) is gathered with periodic
//! wrap of the global indices, so a face block's corner deposits reach the right block
//! across the face after the shift.  The remap is conservative: the sum over a periodic
//! row is preserved, so the total deposited mass is conserved to round-off.
//!
//! The plane cells receive contributions from more than one block (a block's y/z ghost
//! rows overlap its neighbors' active rows), hence the atomic gather and the summing
//! MPI_Allreduce; the summation order can differ with the rank count at the round-off
//! level.  Three device kernels on one execution-space instance (stream ordered); the
//! only explicit fence is the one before the collective.  Only the face blocks take
//! part; with refinement they all sit at shear_lev_, the level of the planes.

TaskStatus MeshBoundaryValuesDep::FoldShearDeposit(DvceArray5D<Real> &a,
                                                   const Real yshear,
                                                   ReconstructionMethod rcon) {
  if (!shear_x1_) {return TaskStatus::complete;}

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, ks = indcs.ks;
  const int nvar = a.extent_int(1);
  const int nmb = pmy_pack->nmb_thispack;
  const int gny = shear_gny_, gnz = shear_gnz_;
  if (nvar != shear_nvar_) {
    Kokkos::realloc(shear_plane_, 2, nvar*ng, gnz, gny);
    shear_nvar_ = nvar;
  }
  auto plane = shear_plane_;
  auto goffs = shear_goffs_;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto &msize = pmy_pack->pmesh->mesh_size;
  const Real ly = msize.x2max - msize.x2min;
  const Real dy = ly/static_cast<Real>(gny);   // global, rank-consistent cell width
  const bool zper = x3_periodic_;

  // 1. gather: x1 ghost slabs of the face blocks (full k,j incl. corners) -> planes
  Kokkos::deep_copy(plane, 0.0);
  par_for("depshear_gather", DevExeSpace(), 0, nmb-1, 0, nx3+2*ng-1, 0, nx2+2*ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    int gj = goffs(m,0) + j - ng;
    gj = (gj % gny + gny) % gny;
    int gk = goffs(m,1) + k - ng;
    if (zper) {
      gk = (gk % gnz + gnz) % gnz;
    } else if ((gk < 0) || (gk >= gnz)) {
      return;   // z ghost rows beyond a non-periodic z face: outside the domain
    }
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      for (int v = 0; v < nvar; ++v) {
        for (int d = 0; d < ng; ++d) {
          Kokkos::atomic_add(&plane(0, v*ng + d, gk, gj), a(m, v, k, j, is - 1 - d));
        }
      }
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      for (int v = 0; v < nvar; ++v) {
        for (int d = 0; d < ng; ++d) {
          Kokkos::atomic_add(&plane(1, v*ng + d, gk, gj), a(m, v, k, j, ie + 1 + d));
        }
      }
    }
  });

#if MPI_PARALLEL_ENABLED
  // reconstruct the global planes on every rank (SUM is the physics here: overlapping
  // ghost rows of neighboring blocks add).  All ranks reach this collective in lockstep.
  Kokkos::fence();
  MPI_Allreduce(MPI_IN_PLACE, plane.data(), static_cast<int>(plane.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // 2. remap each plane row in y (in place: the row is fully loaded into scratch before
  //    the write-back).  Integer shift folded into the wrapped load, fractional part via
  //    the conservative remap fluxes.  Face 0 (inner-ghost deposits) shifts by -yshear,
  //    face 1 (outer-ghost deposits) by +yshear: the adjoint of the ghost-fill signs.
  const int nvd = nvar*ng;
  const int scr_lvl = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(gny + 2*PAD)*2;
  par_for_outer("depshear_remap", DevExeSpace(), scr_size, scr_lvl,
                0, 2*nvd - 1, 0, gnz-1,
  KOKKOS_LAMBDA(TeamMember_t member, const int fd, const int gk) {
    ScrArray1D<Real> q(member.team_scratch(scr_lvl), gny + 2*PAD);
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), gny + 2*PAD);
    const int face = fd / nvd;
    const int vd = fd - face*nvd;
    const Real ysh = (face == 0) ? -yshear : yshear;
    const int joffset = static_cast<int>(ysh/dy);
    const Real eps = fmod(ysh, dy)/dy;
    par_for_inner(member, 0, gny + 2*PAD - 1, [&](const int jf) {
      int jsrc = ((jf - PAD - joffset) % gny + gny) % gny;
      q(jf) = plane(face, vd, gk, jsrc);
    });
    member.team_barrier();
    switch (rcon) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, PAD, PAD+gny, eps, q, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, PAD, PAD+gny, eps, q, flx);
        break;
      default:
        PPMX_RemapFlx(member, PAD, PAD+gny, eps, q, flx);
        break;
    }
    member.team_barrier();
    par_for_inner(member, 0, gny-1, [&](const int j) {
      plane(face, vd, gk, j) = q(j+PAD) - (flx(j+PAD+1) - flx(j+PAD));
    });
  });

  // 3. scatter-add: remapped planes -> active edge strips of the blocks across the face
  //    (inner-ghost deposits land in the OUTER face blocks and vice versa)
  par_for("depshear_scatter", DevExeSpace(), 0, nmb-1, ks, ks+nx3-1, js, js+nx2-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    const int gj = goffs(m,0) + j - js;
    const int gk = goffs(m,1) + k - ks;
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      for (int v = 0; v < nvar; ++v) {
        for (int d = 0; d < ng; ++d) {
          a(m, v, k, j, ie - d) += plane(0, v*ng + d, gk, gj);
        }
      }
    }
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      for (int v = 0; v < nvar; ++v) {
        for (int d = 0; d < ng; ++d) {
          a(m, v, k, j, is + d) += plane(1, v*ng + d, gk, gj);
        }
      }
    }
  });

  return TaskStatus::complete;
}
