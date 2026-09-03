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
//! Only same-level (uniform grid) exchanges are supported.
//!
//! Shear-periodic x1 faces (3D shearing box): the x1 ghost slabs of the face blocks are
//! NOT folded by the plain-periodic pass (their unsheared x1 buffers are skipped in
//! RecvAndSumDeposit); FoldShearDeposit gathers them into global y-z planes, remaps
//! each row in y by -/+yshear with the conservative remap-flux kernels of the orbital
//! advection / multigrid shear machinery (shearing_box/remap_fluxes.hpp), and adds the
//! result into the active edge strips of the blocks across the face.  Sign: the copy
//! exchange fills inner ghosts with outer content shifted by +yshear; the deposit map
//! is its adjoint, so inner-ghost deposits are shifted by -yshear (outer by +yshear).

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
  MeshBoundaryValues(pp, pin, false) {
  Mesh *pm = pp->pmesh;
  if (pm->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Additive deposit exchange does not support SMR/AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // shear-periodic x1 faces: global plane geometry, per-block global offsets, and the
  // x1-direction table of the buffer indices (inverse of NeighborIndex)
  shear_x1_ = (pm->three_d &&
               (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic));
  if (shear_x1_) {
    shear_gny_ = pm->mesh_indcs.nx2;   // uniform grid: root-level global cell counts
    shear_gnz_ = pm->mesh_indcs.nx3;
    x3_periodic_ = (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);
    int nmb = pp->nmb_thispack;
    Kokkos::realloc(shear_goffs_, nmb, 2);
    auto goffs_h = Kokkos::create_mirror_view(shear_goffs_);
    for (int m=0; m<nmb; ++m) {
      LogicalLocation &lloc = pm->lloc_eachmb[m + pp->gids];
      goffs_h(m,0) = static_cast<int>(lloc.lx2)*pm->mb_indcs.nx2;
      goffs_h(m,1) = static_cast<int>(lloc.lx3)*pm->mb_indcs.nx3;
    }
    Kokkos::deep_copy(shear_goffs_, goffs_h);

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
//! \fn void MeshBoundaryValuesDep::InitSendIndices
//! \brief Calculates indices of GHOST cells packed into send buffers. These are the
//! mirror image of the receive ("unpack into ghosts") indices of the copy exchange in
//! MeshBoundaryValuesCC::InitRecvIndices. Only same-level indices are used.

void MeshBoundaryValuesDep::InitSendIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs = pmy_pack->pmesh->mb_indcs;
  int ng = mb_indcs.ng;
  if ((f1 != 0) || (f2 != 0)) {return;}  // only same-level buffers used

  auto &isame = buf.isame[0];
  if (ox1 == 0) {
    isame.bis = mb_indcs.is;          isame.bie = mb_indcs.ie;
  } else if (ox1 > 0) {
    isame.bis = mb_indcs.ie + 1;      isame.bie = mb_indcs.ie + ng;
  } else {
    isame.bis = mb_indcs.is - ng;     isame.bie = mb_indcs.is - 1;
  }

  if (ox2 == 0) {
    isame.bjs = mb_indcs.js;          isame.bje = mb_indcs.je;
  } else if (ox2 > 0) {
    isame.bjs = mb_indcs.je + 1;      isame.bje = mb_indcs.je + ng;
  } else {
    isame.bjs = mb_indcs.js - ng;     isame.bje = mb_indcs.js - 1;
  }

  if (ox3 == 0) {
    isame.bks = mb_indcs.ks;          isame.bke = mb_indcs.ke;
  } else if (ox3 > 0) {
    isame.bks = mb_indcs.ke + 1;      isame.bke = mb_indcs.ke + ng;
  } else {
    isame.bks = mb_indcs.ks - ng;     isame.bke = mb_indcs.ks - 1;
  }
  buf.isame_ndat = (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*
                   (isame.bke - isame.bks + 1);
  // Keep these per-axis extents identical to the partner ranges constructed by
  // InitRecvIndices(). Rank-packed MPI sends and receives must agree on payload size.
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesDep::InitRecvIndices
//! \brief Calculates indices of ACTIVE cells into which receive buffers are summed.
//! These are the mirror image of the send ("pack from active") indices of the copy
//! exchange in MeshBoundaryValuesCC::InitSendIndices.

void MeshBoundaryValuesDep::InitRecvIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs = pmy_pack->pmesh->mb_indcs;
  int ng1 = mb_indcs.ng - 1;
  if ((f1 != 0) || (f2 != 0)) {return;}  // only same-level buffers used

  auto &isame = buf.isame[0];
  isame.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
  isame.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
  isame.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
  isame.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
  isame.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
  isame.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
  buf.isame_ndat = (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*
                   (isame.bke - isame.bks + 1);
  // Keep these per-axis extents identical to the partner ranges constructed by
  // InitSendIndices(). Rank-packed MPI sends and receives must agree on payload size.
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MeshBoundaryValuesDep::PackAndSendDeposit()
//! \brief Pack ghost-region deposits into boundary buffers and send to neighbors.
//! Adapted from MeshBoundaryValuesCC::PackAndSendCC with all coarse/fine/z4c branches
//! removed (uniform grid only). Same-rank neighbors are written directly into the
//! destination receive buffer; the task graph guarantees the receiver does not start
//! summing until all local packs are complete.
//!
//! Input array must be a 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)

TaskStatus MeshBoundaryValuesDep::PackAndSendDeposit(DvceArray5D<Real> &a) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
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
      int il = sbuf[n].isame[0].bis;
      int iu = sbuf[n].isame[0].bie;
      int jl = sbuf[n].isame[0].bjs;
      int ju = sbuf[n].isame[0].bje;
      int kl = sbuf[n].isame[0].bks;
      int ku = sbuf[n].isame[0].bke;
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj  = nk*nj;

      // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
      // in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m,n).dest;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i
        // copy directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
          });
        // else copy into send buffer for MPI communication below
        } else {
#if MPI_PARALLEL_ENABLED
          int base = sendoff(m*nnghbr + n);
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            aggsbuf(base + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = a(m,v,k,j,i);
          });
#else
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
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
//! concurrent accumulation over buffers would race (cf. SumBoundaryFluxes).

TaskStatus MeshBoundaryValuesDep::RecvAndSumDeposit(DvceArray5D<Real> &a) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
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
        int il = rbuf[n].isame[0].bis;
        int iu = rbuf[n].isame[0].bie;
        int jl = rbuf[n].isame[0].bjs;
        int ju = rbuf[n].isame[0].bje;
        int kl = rbuf[n].isame[0].bks;
        int ku = rbuf[n].isame[0].bke;
        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj  = nk*nj;
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

          // Inner (vector) loop over i
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            const int bi = (i-il + ni*(j-jl + nj*(k-kl + nk*v)));
#if MPI_PARALLEL_ENABLED
            a(m,v,k,j,i) += (base >= 0) ? aggrbuf(base + bi) : rbuf[n].vars(m, bi);
#else
            a(m,v,k,j,i) += rbuf[n].vars(m, bi);
#endif
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
//! only explicit fence is the one before the collective.

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
