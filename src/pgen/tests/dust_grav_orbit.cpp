//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_grav_orbit.cpp
//! \brief Dynamic test of the gravitational force on dust particles (dust track, Phase
//! 4c): test particles oscillating vertically in the potential of a self-gravitating
//! isothermal gas slab in hydrostatic equilibrium,
//!     rho(z) = rho0 sech^2(z/h),   h = c_s / sqrt(2 pi G rho0),
//!     phi(z) = 2 c_s^2 ln cosh(z/h),   phi''(0) = 4 pi G rho0.
//! Particles start at rest at height +z0 (or -z0 on ranks owning no MeshBlock at +z0:
//! species 1, the mirrored orbit; all trace the same signed trajectory, and their spread
//! measures any x,y dependence of the discrete force field).  For z0 << h
//! the period is 2 pi / sqrt(4 pi G rho0); for finite z0 the exact period follows from
//! the quadrature of the energy integral (done in the analysis script).  Particles are
//! test particles (<dust>/gravity_source = false, back_reaction = false, a huge stopping
//! time), so the gas evolves as the plain self-gravitating slab.
//! History (user_hist): <z>, <v_z>, rms spread of z over the particles, the particle
//! count, and the gas mass-weighted <z^2> (the slab's own breathing).
//! Parameters: <problem> rho0, z0 (absolute height), floor (density floor / rho0).

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

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

KOKKOS_INLINE_FUNCTION
Real OrbitHash01(std::uint64_t key) {
  key ^= key >> 33; key *= UINT64_C(0xff51afd7ed558ccd);
  key ^= key >> 33; key *= UINT64_C(0xc4ceb9fe1a85ec53);
  key ^= key >> 33;
  return static_cast<Real>(key >> 11)*(1.0/9007199254740992.0);
}

void DustGravOrbitHistory(HistoryData *pdata, Mesh *pm);

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::DustGravOrbit()

void ProblemGenerator::DustGravOrbit(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->pgrav == nullptr || pmbp->ppart == nullptr ||
      pmbp->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust_grav_orbit requires <hydro>, <gravity>, <particles> and <dust>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(pmy_mesh_->three_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust_grav_orbit requires a 3D mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  user_hist_func = DustGravOrbitHistory;

  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  pmbp->pgrav->four_pi_G = four_pi_G;
  if (pmbp->pgrav->pmgd != nullptr) {pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);}
  if (restart) return;

  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real z0 = pin->GetOrAddReal("problem", "z0", 0.05);
  Real floor = pin->GetOrAddReal("problem", "floor", 1.0e-6);
  Real cs = pin->GetReal("hydro", "iso_sound_speed");
  Real h = cs/std::sqrt(0.5*four_pi_G*rho0);     // 2 pi G rho0 = four_pi_G rho0 / 2
  Real omega_z = std::sqrt(four_pi_G*rho0);      // small-amplitude vertical frequency
  if (global_variable::my_rank == 0) {
    std::cout << "# dust_grav_orbit: h = " << h << "  z0/h = " << z0/h
              << "  harmonic period = " << 2.0*M_PI/omega_z << std::endl;
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  bool is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  Real zc = 0.5*(pmy_mesh_->mesh_size.x3min + pmy_mesh_->mesh_size.x3max);

  par_for("orbit_gas", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real sech = 1.0/cosh((z - zc)/h);
    Real rho = fmax(rho0*sech*sech, floor*rho0);
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    if (is_ideal) {u0(m,IEN,k,j,i) = rho*cs*cs/gm1;}
  });

  // particles: at rest at height z0, spread over the (x,y) plane by a hash of the tag
  particles::Particles *ppar = pmbp->ppart;
  int npart = ppar->nprtcl_thispack;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto gids = pmbp->gids;
  auto &taus_ = pmbp->pdust->taus;
  Real mesh_x3min = pmy_mesh_->mesh_size.x3min;
  Real mesh_x3max = pmy_mesh_->mesh_size.x3max;
  if (zc + z0 >= mesh_x3max || zc - z0 <= mesh_x3min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "z0 places the particles outside the mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Each rank places its particles at +z0 (species 0) if it owns a MeshBlock containing
  // that height, else at -z0 (species 1): the two orbits are mirror images, and the
  // history sums z*(1-2*species) so both contribute the same signed trajectory.
  int ncand_host = 0;
  for (int m=0; m<nmb; ++m) {
    if (zc + z0 >= size.h_view(m).x3min && zc + z0 < size.h_view(m).x3max) {
      ++ncand_host;
    }
  }
  int species = (ncand_host > 0) ? 0 : 1;
  Real zp = zc + ((species == 0) ? z0 : -z0);
  if (species == 1) {
    for (int m=0; m<nmb; ++m) {
      if (zp >= size.h_view(m).x3min && zp < size.h_view(m).x3max) {++ncand_host;}
    }
  }
  if (ncand_host == 0 && npart > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rank " << global_variable::my_rank << " owns particles but no MeshBlock "
              << "containing +z0 or -z0; use fewer ranks or a decomposition split in x,y"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int nspec = pmbp->pdust->nspecies;
  if (species == 1 && nspec < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "a rank without +z0 blocks needs <dust>/nspecies = 2 (the -z0 species)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // the owning MeshBlock must contain zp: search each block's z range
  par_for("orbit_part", DevExeSpace(), 0, (npart-1), KOKKOS_LAMBDA(const int p) {
    std::uint64_t key = static_cast<std::uint64_t>(pi(PTAG,p));
    // pick a block containing zp among those on this pack (hash over the candidates)
    int ncand = 0;
    for (int m=0; m<nmb; ++m) {
      if (zp >= size.d_view(m).x3min && zp < size.d_view(m).x3max) {++ncand;}
    }
    int pick = static_cast<int>(OrbitHash01(key)*static_cast<Real>(ncand));
    if (pick > ncand-1) {pick = ncand-1;}
    int m = 0, seen = 0;
    for (int mm=0; mm<nmb; ++mm) {
      if (zp >= size.d_view(mm).x3min && zp < size.d_view(mm).x3max) {
        if (seen == pick) {m = mm;}
        ++seen;
      }
    }
    Real ux = OrbitHash01(key + UINT64_C(0x632be59bd9b4e019));
    Real uy = OrbitHash01(key + UINT64_C(0x8cb92baa3f3d8dd7));
    pr(IPX,p) = size.d_view(m).x1min + ux*(size.d_view(m).x1max - size.d_view(m).x1min);
    pr(IPY,p) = size.d_view(m).x2min + uy*(size.d_view(m).x2max - size.d_view(m).x2min);
    pr(IPZ,p) = zp;
    pi(PGID,p) = gids + m;
    pi(PSP,p) = species;
    pr(IPTS,p) = taus_.d_view(species);
    pr(IPM,p) = 1.0;
    pr(IPVX,p) = 0.0; pr(IPVY,p) = 0.0; pr(IPVZ,p) = 0.0;
    pr(IPRX,p) = 0.0; pr(IPRY,p) = 0.0; pr(IPRZ,p) = 0.0;
  });
  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void DustGravOrbitHistory()

void DustGravOrbitHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  pdata->nhist = 5;
  pdata->label[0] = "zsum";
  pdata->label[1] = "vzsum";
  pdata->label[2] = "z2sum";
  pdata->label[3] = "np";
  pdata->label[4] = "gas_rz2";

  particles::Particles *ppar = pmbp->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  Real zc = 0.5*(pm->mesh_size.x3min + pm->mesh_size.x3max);
  Real sz = 0.0, svz = 0.0, sz2 = 0.0, sn = 0.0;
  Kokkos::parallel_reduce("orbit_hist",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int &p, Real &z_, Real &v_, Real &z2_, Real &n_) {
    Real sgn = (pi(PSP,p) == 0) ? 1.0 : -1.0;   // -z0 species: mirrored orbit
    Real z = sgn*(pr(IPZ,p) - zc);
    z_ += z;
    v_ += sgn*pr(IPVZ,p);
    z2_ += SQR(z);
    n_ += 1.0;
  }, Kokkos::Sum<Real>(sz), Kokkos::Sum<Real>(svz), Kokkos::Sum<Real>(sz2),
     Kokkos::Sum<Real>(sn));

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  Real grz2 = 0.0;
  Kokkos::parallel_reduce("orbit_gas_hist",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &s_) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    s_ += u0(m,IDN,k,j,i)*z*z*vol;
  }, Kokkos::Sum<Real>(grz2));

  pdata->hdata[0] = sz;
  pdata->hdata[1] = svz;
  pdata->hdata[2] = sz2;
  pdata->hdata[3] = sn;
  pdata->hdata[4] = grz2;
  return;
}

}  // namespace
