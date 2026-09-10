//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_amr.cpp
//! \brief Particles on an adaptively refined mesh (dust track): when MeshBlocks are
//! created, destroyed or moved between ranks, every particle is given the GID of the
//! MeshBlock that owns its position on the new tree and, where that block lives on
//! another rank, is sent there through the ordinary particle pipeline.  Optionally the
//! particles of a MeshBlock that refines are first split into 2^d children of equal
//! mass placed half a fine cell from the parent in each direction: a lattice at the
//! coarse cell size becomes exactly the lattice at the fine cell size, so a coarse
//! population entering a finer level keeps a sampling adequate for the finer kernel
//! (the aliasing found with the linA mode on a refined mesh).  Nothing merges particles
//! on derefinement.

#include <algorithm>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn void Particles::RedistributeAfterRemesh()
//! \brief Called from MeshRefinement::RedistAndRefineMeshBlocks after the new tree and
//! load balance are known and refine_flag holds, per OLD MeshBlock, +1 (refined),
//! -nleaf (derefined) or 0, while the Mesh and this pack still carry the old numbering.
//! Collective under MPI.

void Particles::RedistributeAfterRemesh(const int *oldtonew, const int *new_rank_eachmb,
                                        const int new_nmb_total,
                                        const DualArray1D<int> &rflag, const int nleaf) {
  int nmb = pmy_pack->nmb_thispack;
  int gids = pmy_pack->gids;
  DualArray1D<int> o2n("prtcl_o2n", std::max(nmb, 1));
  DualArray1D<int> act("prtcl_act", std::max(nmb, 1));
  for (int m=0; m<nmb; ++m) {
    o2n.h_view(m) = oldtonew[gids + m];
    int f = rflag.h_view(gids + m);
    act.h_view(m) = (f > 0) ? 1 : ((f < 0) ? -1 : 0);
  }
  o2n.template modify<HostMemSpace>();
  o2n.template sync<DevExeSpace>();
  act.template modify<HostMemSpace>();
  act.template sync<DevExeSpace>();
  DualArray1D<int> newrank("prtcl_newrank", std::max(new_nmb_total, 1));
  for (int n=0; n<new_nmb_total; ++n) {newrank.h_view(n) = new_rank_eachmb[n];}
  newrank.template modify<HostMemSpace>();
  newrank.template sync<DevExeSpace>();

  if (split_on_refine) {SplitOnRefine(act, nleaf);}
  pbval_part->RemapAfterRemesh(o2n, act, newrank);
  pbval_part->ExchangeNow();
  // refresh the Mesh particle counts (collective; nothing is dead here)
  (void) RemoveDead();
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::SplitOnRefine()
//! \brief Replaces every particle of a refining MeshBlock (act = +1) by nleaf = 2^d
//! children: child c = i + 2j + 4k sits at the parent position offset by
//! (-1)^(1-i) dx1/4, (-1)^(1-j) dx2/4, (-1)^(1-k) dx3/4 (a quarter of the OLD cell, i.e.
//! half a fine cell), carries the parent's velocity, RK registers (shifted with the
//! position) and drag rates, and 1/nleaf of its mass.  The parent's slot becomes child
//! 0; the others are appended with fresh tags above the global maximum.  Collective.

void Particles::SplitOnRefine(const DualArray1D<int> &act, const int nleaf) {
  int npart = nprtcl_thispack;
  int gids = pmy_pack->gids;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto act_d = act.d_view;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const bool has_reg = (particle_type == ParticleType::dust);

  // parents on this rank, and the largest tag anywhere (children take tags above it)
  int nsplit = 0, maxtag = -1;
  if (npart > 0) {
    Kokkos::parallel_reduce("prtcl_split_count",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &n) {
      if (act_d(pi(PGID,p) - gids) > 0) {++n;}
    }, Kokkos::Sum<int>(nsplit));
    Kokkos::parallel_reduce("prtcl_split_maxtag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &t) {
      t = (pi(PTAG,p) > t) ? pi(PTAG,p) : t;
    }, Kokkos::Max<int>(maxtag));
  }
  int nnew = nsplit*(nleaf - 1);
  std::vector<int> nnew_eachrank(global_variable::nranks, 0);
  nnew_eachrank[global_variable::my_rank] = nnew;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &maxtag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allgather(&nnew, 1, MPI_INT, nnew_eachrank.data(), 1, MPI_INT, MPI_COMM_WORLD);
#endif
  int tag0 = maxtag + 1;
  for (int r=0; r<global_variable::my_rank; ++r) {tag0 += nnew_eachrank[r];}
  if (nnew == 0) {return;}

  // index of each parent among the parents (exclusive scan)
  DvceArray1D<int> map("prtcl_split_map", npart);
  Kokkos::parallel_scan("prtcl_split_scan",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &offset, const bool final) {
    bool parent = (act_d(pi(PGID,p) - gids) > 0);
    if (final) {map(p) = parent ? offset : -1;}
    if (parent) {++offset;}
  });

  int nrd = nrdata, nid = nidata;
  DvceArray2D<Real> new_r("prtcl_rdata", nrd, npart + nnew);
  DvceArray2D<int>  new_i("prtcl_idata", nid, npart + nnew);
  par_for("prtcl_split", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
    for (int n=0; n<nrd; ++n) {new_r(n,p) = pr(n,p);}
    for (int n=0; n<nid; ++n) {new_i(n,p) = pi(n,p);}
    int q = map(p);
    if (q < 0) return;
    int m = pi(PGID,p) - gids;
    Real ox = 0.25*mbsize.d_view(m).dx1;
    Real oy = multi_d ? 0.25*mbsize.d_view(m).dx2 : 0.0;
    Real oz = three_d ? 0.25*mbsize.d_view(m).dx3 : 0.0;
    Real mchild = pr(IPM,p)/static_cast<Real>(nleaf);
    for (int c=0; c<nleaf; ++c) {
      int slot = (c == 0) ? p : npart + q*(nleaf-1) + (c-1);
      if (c > 0) {
        for (int n=0; n<nrd; ++n) {new_r(n,slot) = pr(n,p);}
        for (int n=0; n<nid; ++n) {new_i(n,slot) = pi(n,p);}
        new_i(PTAG,slot) = tag0 + q*(nleaf-1) + (c-1);
      }
      Real sx = (c & 1) ? ox : -ox;
      Real sy = ((c >> 1) & 1) ? oy : -oy;
      Real sz = ((c >> 2) & 1) ? oz : -oz;
      new_r(IPX,slot) += sx;
      new_r(IPY,slot) += sy;
      new_r(IPZ,slot) += sz;
      if (has_reg) {
        new_r(IPX1,slot) += sx;
        new_r(IPY1,slot) += sy;
        new_r(IPZ1,slot) += sz;
      }
      new_r(IPM,slot) = mchild;
    }
  });
  prtcl_rdata = new_r;
  prtcl_idata = new_i;
  nprtcl_thispack = npart + nnew;
}

} // namespace particles
