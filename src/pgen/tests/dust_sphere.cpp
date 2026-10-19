//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_sphere.cpp
//! \brief Pressureless collapse of a uniform sphere of dust particles under their own
//! gravity (Simon et al. 2016, ApJ 822, 55, Sect. 2.2.1; Athena-C prob/particle_sphere.c):
//! a test of the particle self-gravity alone, on uniform and refined meshes.
//!
//! A sphere of radius r0 and particle density rho_p0 is filled with npc^3 particles per
//! cell of every cell whose centre lies inside it (on each MeshBlock at its own level, so a
//! statically refined start keeps the same density), at rest.  The particles source the
//! potential (<dust> gravity_source = true) and feel its force; the gas is a uniform inert
//! background: no drag (huge stopping time), no back-reaction, and no gas self-gravity
//! (<hydro_srcterms> self_gravity off).  With vacuum boundaries (<gravity> mg_bc =
//! multipole) every particle follows the homologous solution
//!     r(t)/r(0) = cos^2(a),   a + sin(2a)/2 = sqrt(8 pi G rho_p0/3) t,
//!     rho_p(t)/rho_p0 = cos^-6(a),   t_ff = sqrt(3 pi/(32 G rho_p0)),
//! so the mass-weighted rms radius and the radius of the outermost particle both shrink by
//! cos^2(a), and the central density rises as cos^-6(a) (until the sphere is a few cells
//! across: the mesh force is softened on ~dx).
//! History (user_hist): particle mass, mass-weighted rms radius about the centre, the
//! largest radius, and the centre of mass.
//! Parameters: <problem> radius, rho_p0, rho_gas, npc (particles per cell per dimension),
//! offset (fixed fractional shift of the lattice within a cell), jitter (random fractional
//! shift amplitude, as particle_sphere.c ipert = 1 with 0.1), xc yc zc, random_seed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "gravity/gravity.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "pgen/pgen.hpp"

namespace {

Real sph_xc = 0.0, sph_yc = 0.0, sph_zc = 0.0;

Real SphereHash01(std::uint64_t key) {
  key ^= key >> 33; key *= UINT64_C(0xff51afd7ed558ccd);
  key ^= key >> 33; key *= UINT64_C(0xc4ceb9fe1a85ec53);
  key ^= key >> 33;
  return static_cast<Real>(key >> 11)*(1.0/9007199254740992.0);
}

void DustSphereHistory(HistoryData *pdata, Mesh *pm);

void Fatal(const char *file, int line, const std::string &msg) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::DustSphere()

void ProblemGenerator::DustSphere(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->pgrav == nullptr || pmbp->ppart == nullptr ||
      pmbp->pdust == nullptr) {
    Fatal(__FILE__, __LINE__, "dust_sphere requires <hydro>, <gravity>, <particles> and <dust>");
  }
  if (!(pmy_mesh_->three_d)) {Fatal(__FILE__, __LINE__, "dust_sphere requires a 3D mesh");}
  user_hist_func = DustSphereHistory;
  sph_xc = pin->GetOrAddReal("problem", "xc", 0.0);
  sph_yc = pin->GetOrAddReal("problem", "yc", 0.0);
  sph_zc = pin->GetOrAddReal("problem", "zc", 0.0);

  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 4.0*M_PI*8.0e-3);
  pmbp->pgrav->four_pi_G = four_pi_G;
  if (pmbp->pgrav->pmgd != nullptr) {pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);}
  if (restart) return;

  Real r0 = pin->GetOrAddReal("problem", "radius", 0.25);
  Real rho_p0 = pin->GetOrAddReal("problem", "rho_p0", 1.0);
  Real rho_gas = pin->GetOrAddReal("problem", "rho_gas", 1.0);
  int npc = pin->GetOrAddInteger("problem", "npc", 1);
  Real offset = pin->GetOrAddReal("problem", "offset", 0.0);
  Real jitter = pin->GetOrAddReal("problem", "jitter", 0.0);
  int64_t seed = pin->GetOrAddInteger("problem", "random_seed", 1);
  Real G = four_pi_G/(4.0*M_PI);
  Real tff = std::sqrt(3.0*M_PI/(32.0*G*rho_p0));

  // gas: uniform and at rest, an inert background
  auto &indcs = pmy_mesh_->mb_indcs;
  int ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng, n2 = indcs.nx2 + 2*ng, n3 = indcs.nx3 + 2*ng;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->phydro->u0;
  bool is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  Real cs = is_ideal ? 1.0 : pmbp->phydro->peos->eos_data.iso_cs;
  par_for("sphere_gas", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = rho_gas;
    u0(m,IM1,k,j,i) = 0.0; u0(m,IM2,k,j,i) = 0.0; u0(m,IM3,k,j,i) = 0.0;
    if (is_ideal) {u0(m,IEN,k,j,i) = rho_gas*cs*cs/gm1;}
  });

  // particles: npc^3 per cell of every cell whose centre is inside the sphere, built on
  // the host (the counts are irregular) and copied to the device
  auto &size = pmbp->pmb->mb_size;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  std::vector<Real> hx, hy, hz, hm;
  std::vector<int> hgid;
  int gids = pmbp->gids;
  int npc3 = npc*npc*npc;
  for (int m=0; m<nmb; ++m) {
    Real x0 = size.h_view(m).x1min, y0 = size.h_view(m).x2min, z0 = size.h_view(m).x3min;
    Real dx = size.h_view(m).dx1, dy = size.h_view(m).dx2, dz = size.h_view(m).dx3;
    Real mp = rho_p0*dx*dy*dz/static_cast<Real>(npc3);
    for (int k=0; k<nx3; ++k) for (int j=0; j<nx2; ++j) for (int i=0; i<nx1; ++i) {
      Real xc = x0 + (i + 0.5)*dx, yc = y0 + (j + 0.5)*dy, zc = z0 + (k + 0.5)*dz;
      if (SQR(xc - sph_xc) + SQR(yc - sph_yc) + SQR(zc - sph_zc) > SQR(r0)) continue;
      for (int kp=0; kp<npc; ++kp) for (int jp=0; jp<npc; ++jp) for (int ip=0; ip<npc; ++ip) {
        std::uint64_t key = (static_cast<std::uint64_t>(gids + m) << 40)
                          ^ (static_cast<std::uint64_t>((k*nx2 + j)*nx1 + i) << 12)
                          ^ static_cast<std::uint64_t>((kp*npc + jp)*npc + ip)
                          ^ static_cast<std::uint64_t>(seed)*UINT64_C(0x9e3779b97f4a7c15);
        Real s[3];
        for (int c=0; c<3; ++c) {
          s[c] = offset + jitter*(SphereHash01(key + UINT64_C(0x632be59bd9b4e019)*(c + 1))
                                  - 0.5);
        }
        hx.push_back(x0 + i*dx + (ip + 0.5 + s[0])*dx/npc);
        hy.push_back(y0 + j*dy + (jp + 0.5 + s[1])*dy/npc);
        hz.push_back(z0 + k*dz + (kp + 0.5 + s[2])*dz/npc);
        hm.push_back(mp);
        hgid.push_back(gids + m);
      }
    }
  }
  int npart = static_cast<int>(hx.size());
  particles::Particles *ppar = pmbp->ppart;
  ppar->nprtcl_thispack = npart;
  Kokkos::realloc(ppar->prtcl_rdata, ppar->nrdata, std::max(npart, 1));
  Kokkos::realloc(ppar->prtcl_idata, ppar->nidata, std::max(npart, 1));
  {
    auto hr = Kokkos::create_mirror_view(ppar->prtcl_rdata);
    auto hi = Kokkos::create_mirror_view(ppar->prtcl_idata);
    Real taus = pmbp->pdust->taus.h_view(0);
    for (int p=0; p<npart; ++p) {
      for (int n=0; n<ppar->nrdata; ++n) {hr(n,p) = 0.0;}
      for (int n=0; n<ppar->nidata; ++n) {hi(n,p) = 0;}
      hr(IPX,p) = hx[p]; hr(IPY,p) = hy[p]; hr(IPZ,p) = hz[p];
      hr(IPTS,p) = taus; hr(IPM,p) = hm[p];
      hi(PGID,p) = hgid[p]; hi(PSP,p) = 0;
    }
    Kokkos::deep_copy(ppar->prtcl_rdata, hr);
    Kokkos::deep_copy(ppar->prtcl_idata, hi);
  }
  Mesh *pm = pmy_mesh_;
  pm->nprtcl_thisrank = npart;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&(pm->nprtcl_thisrank), 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT,
                MPI_COMM_WORLD);
#else
  pm->nprtcl_eachrank[0] = npart;
#endif
  pm->nprtcl_total = 0;
  for (int r=0; r<global_variable::nranks; ++r) {pm->nprtcl_total += pm->nprtcl_eachrank[r];}
  ppar->CreateParticleTags(pin);

  if (global_variable::my_rank == 0) {
    std::cout << "dust_sphere: radius " << r0 << ", rho_p0 " << rho_p0 << ", 4 pi G "
              << four_pi_G << " (G = " << G << "), t_ff = sqrt(3 pi/(32 G rho_p0)) = " << tff
              << ", " << pm->nprtcl_total << " particles (" << npc3 << " per cell), "
              << "sphere mass " << 4.0*M_PI/3.0*r0*r0*r0*rho_p0 << std::endl;
  }
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void DustSphereHistory()

void DustSphereHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  pdata->nhist = 7;
  pdata->label[0] = "mass";
  pdata->label[1] = "mr2";        // sum m r^2 about (xc, yc, zc)
  pdata->label[2] = "rmax";       // largest particle radius (max over ranks)
  pdata->label[3] = "mx";
  pdata->label[4] = "my";
  pdata->label[5] = "mz";
  pdata->label[6] = "np";
  auto &pr = pmbp->ppart->prtcl_rdata;
  int npart = pmbp->ppart->nprtcl_thispack;
  Real xc = sph_xc, yc = sph_yc, zc = sph_zc;
  Real sm = 0.0, smr2 = 0.0, smx = 0.0, smy = 0.0, smz = 0.0, sn = 0.0, rmax = 0.0;
  Kokkos::parallel_reduce("sphere_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int &p, Real &m_, Real &r2_, Real &x_, Real &y_, Real &z_, Real &n_) {
    Real m = pr(IPM,p);
    Real dx = pr(IPX,p) - xc, dy = pr(IPY,p) - yc, dz = pr(IPZ,p) - zc;
    m_ += m; r2_ += m*(dx*dx + dy*dy + dz*dz);
    x_ += m*pr(IPX,p); y_ += m*pr(IPY,p); z_ += m*pr(IPZ,p); n_ += 1.0;
  }, Kokkos::Sum<Real>(sm), Kokkos::Sum<Real>(smr2), Kokkos::Sum<Real>(smx),
     Kokkos::Sum<Real>(smy), Kokkos::Sum<Real>(smz), Kokkos::Sum<Real>(sn));
  Kokkos::parallel_reduce("sphere_rmax", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int &p, Real &r_) {
    Real dx = pr(IPX,p) - xc, dy = pr(IPY,p) - yc, dz = pr(IPZ,p) - zc;
    r_ = fmax(r_, sqrt(dx*dx + dy*dy + dz*dz));
  }, Kokkos::Max<Real>(rmax));
#if MPI_PARALLEL_ENABLED
  // the history sums over ranks; send the rank maximum only from rank 0 after a reduction
  Real rmax_all = rmax;
  MPI_Allreduce(&rmax, &rmax_all, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  rmax = (global_variable::my_rank == 0) ? rmax_all : 0.0;
#endif
  pdata->hdata[0] = sm;
  pdata->hdata[1] = smr2;
  pdata->hdata[2] = rmax;
  pdata->hdata[3] = smx;
  pdata->hdata[4] = smy;
  pdata->hdata[5] = smz;
  pdata->hdata[6] = sn;
}

}  // namespace
