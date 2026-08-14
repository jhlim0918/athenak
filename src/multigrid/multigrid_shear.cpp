//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file multigrid_shear.cpp
//! \brief shear-periodic x1 boundary conditions for the multigrid solver.
//!
//! Shear-periodicity is a boundary identification, not an operator change:
//!   u(x, y, z) = u(x + Lx, y - q*Omega*Lx*t_s, z),
//! so the interior 7-point Laplacian is untouched and only the x1 ghost fill gains a
//! y-shift of +/- qomt*Lx (qomt = q*Omega*t_s, t_s = time since the last strictly
//! shear-periodic instant). The shift is generally a non-integer number of cells at
//! every MG level; the integer part is folded into a periodic-wrapped scratch load and
//! the fractional part is applied with the conservative remap-flux kernels shared with
//! orbital advection and the FFT gravity solver. All sign/index conventions follow the
//! validated fft_gravity.cpp implementation (FillShearGhostPlanes / ComputeQomt).
//!
//! Block levels use two global boundary planes per level (full y-z extent of the level,
//! ngh_ deep in x): gather the interior columns adjacent to each x1 face from the
//! boundary blocks, remap each plane row in y, then scatter into the x1 ghost slabs of
//! the opposite-face blocks -- including the slab's y/z edge and corner ghosts (which
//! trilinear prolongation reads) via periodic wrap in y and z. The root grid is a
//! global array, so it is filled in place (RootShearBoundaryX1).
//!
//! Scope (guarded in MGGravityDriver): 3D uniform grid, root grid on device.
//! Multi-rank: each rank gathers only its own boundary blocks' columns into a
//! zero-filled plane; one MPI_Allreduce(SUM) then reconstructs the global plane on
//! every rank (each cell has exactly one writer, so summing zeros is exact and
//! bit-deterministic), and remap/scatter proceed rank-locally. The root grid is
//! replicated on every rank, so its shear fill needs no communication.

// C++ headers
#include <cmath>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// AthenaK headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../driver/driver.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../shearing_box/remap_fluxes.hpp"
#include "multigrid.hpp"

namespace {
// halo width for the conservative y-remap scratch arrays; PPMX needs j-2..j+2 plus the
// face at ju+1, so 3 is exactly sufficient (integer part of the shift is folded into
// the periodic-wrapped scratch load, so the halo does not depend on the shift size)
constexpr int PAD = 3;
}

//----------------------------------------------------------------------------------------
//! \fn Real MultigridDriver::ComputeShearQomt(Real time)
//! \brief q*Omega*(time since last shear-periodic instant). The box is strictly
//! shear-periodic at times t_n = n*Ly/(q*Omega*Lx); measuring the shift from the most
//! recent t_n keeps the y-offset (and remap error) minimal. Identical to the FFT
//! solver's convention (fft_gravity.cpp) so both solvers agree at any instant.

Real MultigridDriver::ComputeShearQomt(Real time) const {
  if (!mg_shear_enabled_ || mg_qshear_ == 0.0 || mg_omega0_ == 0.0) return 0.0;
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min;
  Real ly = msize.x2max - msize.x2min;
  Real tshear = ly/(mg_qshear_*mg_omega0_*lx);
  Real dts = time - std::floor(time/tshear)*tshear;
  return mg_qshear_*mg_omega0_*dts;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ExecuteMGTaskList(Driver *pdriver, const std::string &tl)
//! \brief run one MG task list to completion. Replicates Driver::ExecuteTaskList but
//! tolerates pdriver == nullptr: every MG task ignores its Driver* argument, which
//! makes static solves from problem generators (constructed before the Driver exists)
//! a supported path.

void MultigridDriver::ExecuteMGTaskList(Driver *pdriver, const std::string &tlname) {
  Mesh *pm = pmy_mesh_;
  MeshBlockPack *pmbp = pm->pmb_pack;
  for (int p = 0; p < (pm->nmb_packs_thisrank); ++p) {
    if (!(pmbp->tl_map[tlname]->Empty())) {pmbp->tl_map[tlname]->Reset();}
  }
  int npack_left = (pm->nmb_packs_thisrank);
  while (npack_left > 0) {
    if (pmbp->tl_map[tlname]->Empty()) {
      npack_left--;
    } else {
      if (!pmbp->tl_map[tlname]->IsComplete()) {
        auto status = pmbp->tl_map[tlname]->DoAvailable(pdriver, 0);
        if (status == TaskListStatus::complete) { npack_left--; }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::AllocateShearPlanes()
//! \brief allocate the per-level global boundary planes and the per-block global
//! (y,z) cell-offset table (uniform grid: all blocks at root level)

void Multigrid::AllocateShearPlanes() {
  if (pmy_pack_ == nullptr) return;  // root grid is filled in place
  // Phase 2b: with boundary annuli the participating blocks (the only ones whose
  // mb_bcs carry shear_periodic on an x1 face) sit at a uniform level above root
  // (guarded in MGGravityDriver); the planes span the y-z extent at THAT level.
  shear_reflev_ = 0;
  for (int m = 0; m < pmy_mesh_->nmb_total; ++m) {
    LogicalLocation &lloc = pmy_mesh_->lloc_eachmb[m];
    std::int32_t nmbx1 =
        (pmy_mesh_->nmb_rootx1 << (lloc.level - pmy_mesh_->root_level));
    if (lloc.lx1 == 0 || lloc.lx1 == (nmbx1-1)) {
      shear_reflev_ = lloc.level - pmy_mesh_->root_level;
      break;  // uniform boundary level: any face block gives the answer
    }
  }
  shear_plane_ = new DvceArray4D<Real>[nlevel_];
  for (int l = 0; l < nlevel_; ++l) {
    int ll = nlevel_ - 1 - l;
    int gny = (nmmbx2_ << shear_reflev_)*(indcs_.nx2 >> ll);
    int gnz = (nmmbx3_ << shear_reflev_)*(indcs_.nx3 >> ll);
    Kokkos::realloc(shear_plane_[l], 2, nvar_*ngh_, gnz, gny);
  }
  Kokkos::realloc(shear_goffs_, nmmb_, 2);
  auto goffs_h = Kokkos::create_mirror_view(shear_goffs_);
  for (int m = 0; m < nmmb_; ++m) {
    LogicalLocation &lloc = pmy_mesh_->lloc_eachmb[m + pmy_pack_->gids];
    goffs_h(m,0) = static_cast<int>(lloc.lx2)*indcs_.nx2;
    goffs_h(m,1) = static_cast<int>(lloc.lx3)*indcs_.nx3;
  }
  Kokkos::deep_copy(shear_goffs_, goffs_h);
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::FillShearGhostPack(Real qomt, ReconstructionMethod order)
//! \brief fill the x1 ghost slabs of x1-boundary blocks at the current MG level with
//! the sheared image of the opposite boundary. Sign convention (validated in
//! fft_gravity.cpp): inner ghosts = outer-boundary content shifted by +qomt*Lx,
//! outer ghosts = inner content shifted by -qomt*Lx.
//!
//! Three kernels on one execution-space instance (stream-ordered, so no fences):
//! gather reads only interior cells, remap works within the plane, scatter writes only
//! ghost cells; the whole slab k,j in [0, ncells+2*ngh) is written, with periodic wrap
//! in y and z supplying the slab's edge/corner ghosts.

void Multigrid::FillShearGhostPack(Real qomt, ReconstructionMethod order) {
  if (shear_plane_ == nullptr) return;
  int ll = nlevel_ - 1 - current_level_;
  int ncells = indcs_.nx1 >> ll;
  if (ncells < 1) return;
  int gny = (nmmbx2_ << shear_reflev_)*(indcs_.nx2 >> ll);
  int gnz = (nmmbx3_ << shear_reflev_)*(indcs_.nx3 >> ll);
  int ngh = ngh_, nvar = nvar_, nmb = nmmb_;
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min;
  Real ly = msize.x2max - msize.x2min;
  Real dy_l = ly/static_cast<Real>(gny);
  Real ysh = qomt*lx;
  auto u = u_[current_level_].d_view;
  auto plane = shear_plane_[current_level_];
  auto goffs = shear_goffs_;
  auto &mb_bcs = pmy_pack_->pmb->mb_bcs;

  // 1. gather: interior columns adjacent to each x1 face -> global planes.
  //    plane face 0 (source for inner ghosts) <- columns at the outer boundary;
  //    plane face 1 (source for outer ghosts) <- columns at the inner boundary.
#if MPI_PARALLEL_ENABLED
  // each rank contributes only its own boundary blocks; zero first, sum after
  Kokkos::deep_copy(plane, 0.0);
#endif
  par_for("mgshear_gather", DevExeSpace(), 0, nmb-1, ngh, ngh+ncells-1, ngh,
          ngh+ncells-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    int gj = (goffs(m,0) >> ll) + (j - ngh);
    int gk = (goffs(m,1) >> ll) + (k - ngh);
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      for (int v = 0; v < nvar; ++v) {
        for (int d = 0; d < ngh; ++d) {
          plane(0, v*ngh + d, gk, gj) = u(m, v, k, j, ngh + ncells - 1 - d);
        }
      }
    }
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      for (int v = 0; v < nvar; ++v) {
        for (int d = 0; d < ngh; ++d) {
          plane(1, v*ngh + d, gk, gj) = u(m, v, k, j, ngh + d);
        }
      }
    }
  });

#if MPI_PARALLEL_ENABLED
  // reconstruct the global plane on every rank: every cell was written by exactly
  // one rank (boundary blocks tile the level's y-z extent), rest are zero, so the
  // sum is exact and independent of reduction order. All ranks reach this collective
  // in lockstep: the call sites skip it under level-global conditions only.
  Kokkos::fence();
  MPI_Allreduce(MPI_IN_PLACE, plane.data(), static_cast<int>(plane.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // 2. remap each plane row in y (in place: the row is fully loaded into scratch
  //    before the write-back). Integer shift folded into the wrapped load;
  //    fractional part via the conservative remap fluxes.
  int nvd = nvar*ngh;
  int scr_lvl = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(gny + 2*PAD)*2;
  par_for_outer("mgshear_remap", DevExeSpace(), scr_size, scr_lvl,
                0, 2*nvd - 1, 0, gnz-1,
  KOKKOS_LAMBDA(TeamMember_t member, const int fd, const int gk) {
    ScrArray1D<Real> q(member.team_scratch(scr_lvl), gny + 2*PAD);
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), gny + 2*PAD);
    int face = fd / nvd;
    int vd = fd - face*nvd;
    Real yshear = (face == 0) ? ysh : -ysh;
    int joffset = static_cast<int>(yshear/dy_l);
    Real eps = fmod(yshear, dy_l)/dy_l;
    par_for_inner(member, 0, gny + 2*PAD - 1, [&](const int jf) {
      int jsrc = ((jf - PAD - joffset) % gny + gny) % gny;
      q(jf) = plane(face, vd, gk, jsrc);
    });
    member.team_barrier();
    switch (order) {
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

  // 3. scatter: remapped planes -> x1 ghost slabs (full k,j range incl. corners)
  par_for("mgshear_scatter", DevExeSpace(), 0, nmb-1, 0, ncells+2*ngh-1, 0,
          ncells+2*ngh-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    int gj = ((goffs(m,0) >> ll) + j - ngh);
    int gk = ((goffs(m,1) >> ll) + k - ngh);
    gj = (gj % gny + gny) % gny;
    gk = (gk % gnz + gnz) % gnz;
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      for (int v = 0; v < nvar; ++v) {
        for (int d = 0; d < ngh; ++d) {
          u(m, v, k, j, ngh - 1 - d) = plane(0, v*ngh + d, gk, gj);
        }
      }
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      for (int v = 0; v < nvar; ++v) {
        for (int d = 0; d < ngh; ++d) {
          u(m, v, k, j, ngh + ncells + d) = plane(1, v*ngh + d, gk, gj);
        }
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::FillShearOctetGhosts(int lev, bool folddata)
//! \brief shear-periodic x1 ghost fill for the octets of refinement level lev (host
//! code: octet buffers live on the host and are globally replicated on every rank, so
//! no communication is needed). The Phase-1 global-plane algorithm at octet
//! granularity: gather the interior cell columns adjacent to each x1 face from the
//! boundary octets (which tile the full y-z extent under the uniform-boundary-level
//! policy), remap each row in y with the scalar twins of the conservative remap
//! kernels, and scatter into the x1 ghost slabs of the opposite-face octets --
//! including the slab's y/z edge and corner ghosts (trilinear prolongation and the
//! flux-conserving boundary prolongation read those), which overwrites the unsheared
//! periodic values the neighbor-table exchange just wrote. Sign convention identical
//! to FillShearGhostPack: inner ghosts = outer-boundary content shifted by +qomt*Lx,
//! outer ghosts = inner content shifted by -qomt*Lx. Assumes mg_nghost == 1 (guarded
//! in MGGravityDriver): the plane is one cell deep and octet interiors are 2 cells.
//! Coarse-level ghost quality only affects the convergence rate -- the V-cycle fixed
//! point is set by the finest block level (FillShearGhostPack) -- but a consistent
//! fill keeps the sheared contraction at its periodic value.

void MultigridDriver::FillShearOctetGhosts(int lev, bool folddata) {
  if (!mg_shear_enabled_ || mg_qomt_ == 0.0) return;
  if (!oct_shear_face_[lev]) return;
  const int ngh = mgroot_->ngh_;
  const int maxlx1 = nrbx1_ << lev;
  const int gny = (nrbx2_ << lev) * 2;
  const int gnz = (nrbx3_ << lev) * 2;
  const int nc = 2 + 2*ngh;
  auto &msize = pmy_mesh_->mesh_size;
  const Real lx = msize.x1max - msize.x1min;
  const Real ly = msize.x2max - msize.x2min;
  const Real dy_l = ly/static_cast<Real>(gny);
  const int nfield = folddata ? 2 : 1;

  // plane(face, field, v, gk, gj); face 0 = source for inner ghosts (gathered at the
  // outer boundary), face 1 = source for outer ghosts (gathered at the inner boundary)
  std::vector<Real> plane(static_cast<std::size_t>(2)*nfield*nvar_*gnz*gny, 0.0);
  auto P = [&](int f, int fd, int v, int gk, int gj) -> Real& {
    return plane[(((static_cast<std::size_t>(f)*nfield + fd)*nvar_ + v)*gnz + gk)*gny
                 + gj];
  };

  // 1. gather interior columns adjacent to each x1 face
  for (int o = 0; o < noctets_[lev]; ++o) {
    MGOctet &oct = octets_[lev][o];
    int fx, isrc;
    if (oct.loc.lx1 == maxlx1-1)   { fx = 0; isrc = ngh + 1; }
    else if (oct.loc.lx1 == 0)     { fx = 1; isrc = ngh; }
    else continue;
    const int gj0 = 2*static_cast<int>(oct.loc.lx2);
    const int gk0 = 2*static_cast<int>(oct.loc.lx3);
    for (int v = 0; v < nvar_; ++v) {
      for (int k = 0; k <= 1; ++k) {
        for (int j = 0; j <= 1; ++j) {
          P(fx, 0, v, gk0+k, gj0+j) = oct.U(v, ngh+k, ngh+j, isrc);
          if (folddata) P(fx, 1, v, gk0+k, gj0+j) = oct.Uold(v, ngh+k, ngh+j, isrc);
        }
      }
    }
  }

  // 2. remap each row in y: integer part of the shift folded into the wrapped scratch
  //    load, fractional part via the conservative remap fluxes (scalar face twins)
  std::vector<Real> q(gny + 2*PAD), flx(gny + 2*PAD);
  for (int f = 0; f < 2; ++f) {
    const Real ysh = (f == 0) ? mg_qomt_*lx : -mg_qomt_*lx;
    const int joffset = static_cast<int>(ysh/dy_l);
    const Real eps = std::fmod(ysh, dy_l)/dy_l;
    for (int fd = 0; fd < nfield; ++fd) {
      for (int v = 0; v < nvar_; ++v) {
        for (int gk = 0; gk < gnz; ++gk) {
          for (int jf = 0; jf < gny + 2*PAD; ++jf) {
            int jsrc = ((jf - PAD - joffset) % gny + gny) % gny;
            q[jf] = P(f, fd, v, gk, jsrc);
          }
          for (int jf = PAD; jf <= PAD + gny; ++jf) {
            switch (mg_remap_order_) {
              case ReconstructionMethod::dc:
                flx[jf] = DC_RemapFlxFace(q.data(), jf, eps);
                break;
              case ReconstructionMethod::plm:
                flx[jf] = PLM_RemapFlxFace(q.data(), jf, eps);
                break;
              default:
                flx[jf] = PPMX_RemapFlxFace(q.data(), jf, eps);
                break;
            }
          }
          for (int j = 0; j < gny; ++j) {
            P(f, fd, v, gk, j) = q[j+PAD] - (flx[j+PAD+1] - flx[j+PAD]);
          }
        }
      }
    }
  }

  // 3. scatter into the x1 ghost slabs (full j,k range incl. corners, periodic wrap)
  for (int o = 0; o < noctets_[lev]; ++o) {
    MGOctet &oct = octets_[lev][o];
    const bool inner = (oct.loc.lx1 == 0);
    const bool outer = (oct.loc.lx1 == maxlx1-1);
    if (!inner && !outer) continue;
    const int gj0 = 2*static_cast<int>(oct.loc.lx2);
    const int gk0 = 2*static_cast<int>(oct.loc.lx3);
    for (int k = 0; k < nc; ++k) {
      int gk = ((gk0 + k - ngh) % gnz + gnz) % gnz;
      for (int j = 0; j < nc; ++j) {
        int gj = ((gj0 + j - ngh) % gny + gny) % gny;
        for (int v = 0; v < nvar_; ++v) {
          if (inner) {
            oct.U(v, k, j, ngh-1) = P(0, 0, v, gk, gj);
            if (folddata) oct.Uold(v, k, j, ngh-1) = P(0, 1, v, gk, gj);
          }
          if (outer) {
            oct.U(v, k, j, ngh+2) = P(1, 0, v, gk, gj);
            if (folddata) oct.Uold(v, k, j, ngh+2) = P(1, 1, v, gk, gj);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::RootShearBoundaryX1()
//! \brief shear-periodic x1 ghost fill of the (global, replicated) root grid at its
//! current level, device path only (root_on_host + shear is guarded fatal). Fills the
//! interior-row ghost columns; the x2/x3 passes of MGRootBoundary then supply the
//! ghost-corner rows by periodic wrap of these remapped rows. Reads (interior columns)
//! and writes (ghost columns) do not overlap. Reduces to the plain periodic copy when
//! qomt == 0 (remap fluxes vanish identically at eps == 0).

void MultigridDriver::RootShearBoundaryX1() {
  int lev = mgroot_->GetCurrentLevel();
  int ll = mgroot_->GetNumberOfLevels() - 1 - lev;
  int ngh = mgroot_->ngh_;
  int ncx = mgroot_->indcs_.nx1 >> ll;
  int ncy = mgroot_->indcs_.nx2 >> ll;
  int ncz = mgroot_->indcs_.nx3 >> ll;
  int nvar = nvar_;
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min;
  Real dy_l = (msize.x2max - msize.x2min)/static_cast<Real>(ncy);
  Real ysh = mg_qomt_*lx;
  auto order = mg_remap_order_;
  auto u = mgroot_->GetCurrentData();

  int nvd = nvar*ngh;
  int scr_lvl = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncy + 2*PAD)*2;
  par_for_outer("mgshear_root", DevExeSpace(), scr_size, scr_lvl,
                0, 2*nvd - 1, ngh, ngh+ncz-1,
  KOKKOS_LAMBDA(TeamMember_t member, const int fd, const int k) {
    ScrArray1D<Real> q(member.team_scratch(scr_lvl), ncy + 2*PAD);
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), ncy + 2*PAD);
    int face = fd / nvd;
    int vd = fd - face*nvd;
    int v = vd / ngh;
    int d = vd - v*ngh;
    // face 0: inner ghost i = ngh-1-d <- interior column next to the outer face,
    //         content shifted by +qomt*Lx; face 1: mirrored, shifted by -qomt*Lx
    Real yshear = (face == 0) ? ysh : -ysh;
    int isrc = (face == 0) ? (ngh + ncx - 1 - d) : (ngh + d);
    int idst = (face == 0) ? (ngh - 1 - d) : (ngh + ncx + d);
    int joffset = static_cast<int>(yshear/dy_l);
    Real eps = fmod(yshear, dy_l)/dy_l;
    par_for_inner(member, 0, ncy + 2*PAD - 1, [&](const int jf) {
      int jsrc = ((jf - PAD - joffset) % ncy + ncy) % ncy;
      q(jf) = u(0, v, k, ngh + jsrc, isrc);
    });
    member.team_barrier();
    switch (order) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, PAD, PAD+ncy, eps, q, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, PAD, PAD+ncy, eps, q, flx);
        break;
      default:
        PPMX_RemapFlx(member, PAD, PAD+ncy, eps, q, flx);
        break;
    }
    member.team_barrier();
    par_for_inner(member, 0, ncy-1, [&](const int j) {
      u(0, v, k, ngh + j, idst) = q(j+PAD) - (flx(j+PAD+1) - flx(j+PAD));
    });
  });
}
