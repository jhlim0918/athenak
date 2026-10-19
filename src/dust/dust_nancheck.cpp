//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_nancheck.cpp
//! \brief Debug tripwire for the dust module (<dust>/nan_check = true): after each guarded
//! task (NanCheckGuard) scan the active cells of the gas state, the dust deposit and drag
//! fields and the gravity force, and every particle, for non-finite values or a
//! non-positive gas density.  At the first hit on any rank, print where (task, stage, cycle,
//! time), how many entries are bad in each array, and the full state of the first bad cell
//! and the first bad particle, then MPI_Abort.  Written to find the source of the NaN
//! particle positions that ended the NHFP self-gravity run on Vista and the three-level
//! box02 branch on NAS (2026-10-16): both died with NaN positions on every rank.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"

namespace dust {

namespace {

// number of bad entries in the active cells of a (nmb, nvar, n3, n2, n1) array and the
// linear index (m, v, k, j, i) of the first one; positive = also flag a(m,0,...) <= 0
struct BadField {int64_t n = 0; int64_t first = std::numeric_limits<int64_t>::max();};

BadField ScanField(const DvceArray5D<Real> &a, int nmb, int nvar, const RegionIndcs &in,
                   bool positive_first) {
  BadField r;
  if (a.extent_int(0) < nmb || a.extent_int(1) < nvar || nvar <= 0) {return r;}
  const int is = in.is, js = in.js, ks = in.ks, nx1 = in.nx1, nx2 = in.nx2, nx3 = in.nx3;
  const int64_t nk = nx3, nj = nx2, ni = nx1, nv = nvar;
  const int64_t ntot = static_cast<int64_t>(nmb)*nv*nk*nj*ni;
  int64_t nbad = 0, first = std::numeric_limits<int64_t>::max();
  Kokkos::parallel_reduce("dust_nan_field", Kokkos::RangePolicy<DevExeSpace, Kokkos::IndexType<int64_t>>(0, ntot),
  KOKKOS_LAMBDA(const int64_t idx, int64_t &cnt, int64_t &fst) {
    int64_t q = idx;
    const int i = static_cast<int>(q % ni); q /= ni;
    const int j = static_cast<int>(q % nj); q /= nj;
    const int k = static_cast<int>(q % nk); q /= nk;
    const int v = static_cast<int>(q % nv); q /= nv;
    const int m = static_cast<int>(q);
    const Real x = a(m, v, k + ks, j + js, i + is);
    bool bad = !Kokkos::isfinite(x) || (positive_first && v == 0 && !(x > 0.0));
    if (bad) {cnt += 1; if (idx < fst) {fst = idx;}}
  }, Kokkos::Sum<int64_t>(nbad), Kokkos::Min<int64_t>(first));
  r.n = nbad; r.first = first;
  return r;
}

void Unflatten(int64_t idx, int nvar, const RegionIndcs &in, int &m, int &v, int &k, int &j,
               int &i) {
  i = static_cast<int>(idx % in.nx1); idx /= in.nx1;
  j = static_cast<int>(idx % in.nx2); idx /= in.nx2;
  k = static_cast<int>(idx % in.nx3); idx /= in.nx3;
  v = static_cast<int>(idx % nvar); idx /= nvar;
  m = static_cast<int>(idx);
  i += in.is; j += in.js; k += in.ks;
}

}  // namespace

void DustGasDrag::NanCheck(const char *where, int stage) {
  MeshBlockPack *pp = pmy_pack;
  Mesh *pm = pp->pmesh;
  auto &in = pm->mb_indcs;
  const int nmb = pp->nmb_thispack;
  struct Item {const char *name; const DvceArray5D<Real> *a; int nvar; bool pos;};
  Item items[6];
  int nitem = 0;
  hydro::Hydro *phyd = pp->phydro;
  if (phyd != nullptr) {items[nitem++] = {"gas u0", &phyd->u0, phyd->u0.extent_int(1), true};}
  items[nitem++] = {"qdep", &qdep, qdep.extent_int(1), false};
  items[nitem++] = {"ustar", &ustar, ustar.extent_int(1), false};
  items[nitem++] = {"dmom", &dmom, dmom.extent_int(1), false};
  if (gravity) {
    items[nitem++] = {"rho_dust", &rho_dust, rho_dust.extent_int(1), false};
    items[nitem++] = {"gforce", &gforce, gforce.extent_int(1), false};
  }
  BadField bf[6];
  bool anybad = false;
  for (int n=0; n<nitem; ++n) {
    bf[n] = ScanField(*items[n].a, nmb, items[n].nvar, in, items[n].pos);
    if (bf[n].n > 0) {anybad = true;}
  }
  // particles: position, velocity and mass
  particles::Particles *ppar = pp->ppart;
  int64_t pbad = 0, pfirst = std::numeric_limits<int64_t>::max();
  int npart = (ppar != nullptr) ? ppar->nprtcl_thispack : 0;
  if (npart > 0) {
    auto &pr = ppar->prtcl_rdata;
    Kokkos::parallel_reduce("dust_nan_part", Kokkos::RangePolicy<DevExeSpace, Kokkos::IndexType<int64_t>>(0, npart),
    KOKKOS_LAMBDA(const int64_t p, int64_t &cnt, int64_t &fst) {
      bool bad = !Kokkos::isfinite(pr(IPX,p)) || !Kokkos::isfinite(pr(IPY,p)) || !Kokkos::isfinite(pr(IPZ,p)) ||
                 !Kokkos::isfinite(pr(IPVX,p)) || !Kokkos::isfinite(pr(IPVY,p)) || !Kokkos::isfinite(pr(IPVZ,p)) ||
                 !(pr(IPM,p) > 0.0) || !Kokkos::isfinite(pr(IPM,p));
      if (bad) {cnt += 1; if (p < fst) {fst = p;}}
    }, Kokkos::Sum<int64_t>(pbad), Kokkos::Min<int64_t>(pfirst));
    if (pbad > 0) {anybad = true;}
  }
  if (!anybad) {return;}

  // report (this rank), then stop every rank
  const int rank = global_variable::my_rank;
  std::printf("### DUST NAN_CHECK rank %d: bad values after %s, stage %d, cycle %d, time %.15g\n",
              rank, where, stage, pm->ncycle, pm->time);
  auto gid_h = pp->pmb->mb_gid.h_view;
  auto &size = pp->pmb->mb_size;
  for (int n=0; n<nitem; ++n) {
    if (bf[n].n == 0) {continue;}
    int m, v, k, j, i;
    Unflatten(bf[n].first, items[n].nvar, in, m, v, k, j, i);
    auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), *items[n].a);
    Real x1 = size.h_view(m).x1min + (i - in.is + 0.5)*size.h_view(m).dx1;
    Real x2 = size.h_view(m).x2min + (j - in.js + 0.5)*size.h_view(m).dx2;
    Real x3 = size.h_view(m).x3min + (k - in.ks + 0.5)*size.h_view(m).dx3;
    std::printf("###   %-8s: %lld bad entries; first at gid %d (level %d) var %d cell (%d,%d,%d) "
                "x = (%.6e, %.6e, %.6e):", items[n].name, static_cast<long long>(bf[n].n),
                gid_h(m), pp->pmb->mb_lev.h_view(m), v, i, j, k, x1, x2, x3);
    for (int vv=0; vv<items[n].nvar; ++vv) {std::printf(" %.6e", h(m, vv, k, j, i));}
    std::printf("\n");
    if (phyd != nullptr && n > 0) {      // the gas in the same cell
      auto hu = Kokkos::create_mirror_view_and_copy(HostMemSpace(), phyd->u0);
      std::printf("###            gas there:");
      for (int vv=0; vv<phyd->u0.extent_int(1); ++vv) {std::printf(" %.6e", hu(m, vv, k, j, i));}
      std::printf("\n");
    }
  }
  if (pbad > 0) {
    auto hr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppar->prtcl_rdata);
    auto hi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppar->prtcl_idata);
    const int p = static_cast<int>(pfirst);
    std::printf("###   particles: %lld of %d bad; first p = %d gid %d tag %d species %d plev %d\n",
                static_cast<long long>(pbad), npart, p, hi(PGID,p), hi(PTAG,p), hi(PSP,p), hi(PLEV,p));
    std::printf("###     rdata:");
    for (int n=0; n<ppar->nrdata; ++n) {std::printf(" %.6e", hr(n,p));}
    std::printf("\n");
  }
  std::fflush(stdout);
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, 3);
#endif
  std::exit(3);
}

}  // namespace dust
