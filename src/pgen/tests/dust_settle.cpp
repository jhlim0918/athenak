//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_settle.cpp
//! \brief Tide + drag analytics for test particles in the stratified 3D shearing box
//! (dust track, particles + SMR ladder, step 1).  No self-gravity.  The gas is an
//! isothermal slab in hydrostatic equilibrium with the vertical tide,
//!     rho(z) = rho0 exp(-z^2/(2 H^2)),   H = c_s/Omega,
//! at the dust-free NSH state u = (0, -etavk, 0) (shear-relative), balanced by the radial
//! pressure-gradient mimic <hydro_srcterms> const_accel = 2 Omega etavk.  That
//! horizontal state is an exact discrete fixed point (uniform in x,y,z), so the gas
//! velocity seen by the particles is (0, -etavk, w) with w the hydrostatic truncation
//! residual.  Test particles (back_reaction = false) of nspecies stopping times start at
//! height z0, spread over (x,y) by a hash of their tag inside the MeshBlocks containing
//! z0, at the test-particle NSH velocities
//!     v_x = -2 Omega tau etavk / D,  v_phi = -etavk / D,  D = 1 + 2(2-q) Omega^2 tau^2,
//! and v_z = 0.  Analytic references:
//!   * horizontal: the NSH velocities are held to round-off (the gas state is exact);
//!   * vertical: z'' + z'/tau + Omega^2 z = 0 with z(0) = z0, z'(0) = 0, i.e. with
//!     lambda = 1/(2 tau): Omega > lambda: z = z0 e^{-lambda t}(cos wt + lambda/w sin wt)
//!     w = sqrt(Omega^2 - lambda^2) (damped vertical epicycle);  Omega < lambda:
//!     cosh/sinh with k = sqrt(lambda^2 - Omega^2) (overdamped settling, terminal rate
//!     Omega^2 tau for Omega tau << 1).
//! History (user_hist, per species s): zin_s nin_s zout_s nout_s zex_s = the sums of z
//! and the counts of the particles with |x| < <problem>/xsplit ("in") and |x| >=
//! xsplit ("out"), and the analytic z(t).  With a refined bar |x| < xsplit the two
//! populations settle through fine resp. root MeshBlocks of the same run.  Per-species
//! velocity means come from the phst output.
//! Parameters: <problem> rho0, etavk, z0, floor, xsplit (<= 0: all particles "in").
//! Requires: 3D, <shearing_box> with stratified = true, <particles>, <dust> with
//! back_reaction = false and nspecies <= 3, isothermal or ideal <hydro>.

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
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "pgen/pgen.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
Real SettleHash01(std::uint64_t key) {
  key ^= key >> 33; key *= UINT64_C(0xff51afd7ed558ccd);
  key ^= key >> 33; key *= UINT64_C(0xc4ceb9fe1a85ec53);
  key ^= key >> 33;
  return static_cast<Real>(key >> 11)*(1.0/9007199254740992.0);
}

// parameters shared with the history function
Real settle_z0 = 0.0, settle_omega = 1.0, settle_xsplit = 0.0;

//----------------------------------------------------------------------------------------
//! \fn Real SettleExact
//! \brief z(t) of the damped vertical oscillator z'' + z'/tau + Omega^2 z = 0 released
//! at rest from z0

Real SettleExact(const Real t, const Real tau, const Real omega, const Real z0) {
  const Real lam = 0.5/tau;
  const Real e = std::exp(-lam*t);
  if (omega > lam) {
    const Real w = std::sqrt(omega*omega - lam*lam);
    return z0*e*(std::cos(w*t) + (lam/w)*std::sin(w*t));
  } else if (omega < lam) {
    const Real k = std::sqrt(lam*lam - omega*omega);
    return z0*e*(std::cosh(k*t) + (lam/k)*std::sinh(k*t));
  }
  return z0*e*(1.0 + lam*t);   // critically damped
}

void DustSettleHistory(HistoryData *pdata, Mesh *pm);

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::DustSettle()

void ProblemGenerator::DustSettle(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust_settle requires <hydro>, <particles> and <dust>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(pmy_mesh_->three_d) || !(pin->DoesBlockExist("shearing_box")) ||
      !(pin->GetOrAddBoolean("shearing_box","stratified",false))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust_settle requires a 3D mesh and <shearing_box> with stratified=true"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pdust->back_reaction) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust_settle is a test-particle problem: set <dust>/back_reaction=false"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nspec = pmbp->pdust->nspecies;
  if (nspec > 3) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust_settle history supports at most 3 species" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  user_hist_func = DustSettleHistory;

  const Real omega0 = pin->GetReal("shearing_box","omega0");
  const Real qshear = pin->GetReal("shearing_box","qshear");
  const Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  const Real etavk = pin->GetOrAddReal("problem","etavk",0.05);
  const Real z0 = pin->GetOrAddReal("problem","z0",1.5);
  const Real floor = pin->GetOrAddReal("problem","floor",1.0e-6);
  settle_xsplit = pin->GetOrAddReal("problem","xsplit",0.0);
  settle_z0 = z0;
  settle_omega = omega0;
  if (restart) return;

  // the radial forcing must be applied to the gas by the standard const_accel source
  // term (so that dust particles do not feel it); verify the input file sets it up
  {
    const Real ax = 2.0*omega0*etavk;
    bool ok = pin->GetOrAddBoolean("hydro_srcterms","const_accel",false);
    Real val = ok ? pin->GetReal("hydro_srcterms","const_accel_val") : 0.0;
    int dir = ok ? pin->GetInteger("hydro_srcterms","const_accel_dir") : 0;
    bool need = (etavk != 0.0);
    if ((need && (!ok || dir != 1 || std::fabs(val - ax) > 1.0e-12*std::fabs(ax))) ||
        (!need && ok && val != 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "dust_settle requires <hydro_srcterms> const_accel=true, "
                << "const_accel_dir = 1, const_accel_val = 2*omega0*etavk = " << ax
                << " (or no forcing when etavk = 0)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  const bool is_ideal = eos.is_ideal;
  const Real cs = is_ideal ? std::sqrt(pin->GetReal("problem","pgas")/rho0)
                           : eos.iso_cs;
  const Real gm1 = is_ideal ? (eos.gamma - 1.0) : 1.0;
  const Real hgas = cs/omega0;
  const Real zc = 0.5*(pmy_mesh_->mesh_size.x3min + pmy_mesh_->mesh_size.x3max);
  if (global_variable::my_rank == 0) {
    std::cout << "# dust_settle: H = " << hgas << "  z0/H = " << z0/hgas
              << "  Omega = " << omega0 << "  etavk = " << etavk << std::endl;
    for (int s=0; s<nspec; ++s) {
      Real tau = pmbp->pdust->taus.h_view(s);
      Real d = 1.0 + 2.0*(2.0-qshear)*SQR(omega0*tau);
      std::cout << "#   species " << s+1 << ": tau = " << tau
                << "  Omega*tau = " << omega0*tau
                << "  NSH v_x = " << -2.0*omega0*tau*etavk/d
                << "  v_phi = " << -etavk/d
                << (omega0*tau < 0.5 ? "  (overdamped settling)" : "  (damped epicycle)")
                << std::endl;
    }
  }

  // gas: hydrostatic isothermal slab at the dust-free NSH horizontal state
  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng, n2 = indcs.nx2 + 2*ng, n3 = indcs.nx3 + 2*ng;
  const int ks = indcs.ks;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  par_for("settle_gas", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rho = fmax(rho0*exp(-0.5*SQR((z - zc)/hgas)), floor*rho0);
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = -rho*etavk;
    u0(m,IM3,k,j,i) = 0.0;
    if (is_ideal) {u0(m,IEN,k,j,i) = rho*cs*cs/gm1 + 0.5*rho*SQR(etavk);}
  });

  // particles: at rest vertically at height z0, spread over the (x,y) plane of the
  // MeshBlocks of this rank that contain z0 (hash of the tag), at the NSH velocities
  particles::Particles *ppar = pmbp->ppart;
  int npart = ppar->nprtcl_thispack;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto gids = pmbp->gids;
  auto &taus_ = pmbp->pdust->taus;
  const Real zp = zc + z0;
  if (zp >= pmy_mesh_->mesh_size.x3max || zp <= pmy_mesh_->mesh_size.x3min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "z0 places the particles outside the mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int ncand_host = 0;
  for (int m=0; m<nmb; ++m) {
    if (zp >= size.h_view(m).x3min && zp < size.h_view(m).x3max) {++ncand_host;}
  }
  if (ncand_host == 0 && npart > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rank " << global_variable::my_rank << " owns particles but no block "
              << "containing z0; use fewer ranks or a decomposition split in x,y"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real kap = 2.0*(2.0-qshear);
  par_for("settle_part", DevExeSpace(), 0, (npart-1), KOKKOS_LAMBDA(const int p) {
    std::uint64_t key = static_cast<std::uint64_t>(pi(PTAG,p));
    int ncand = 0;
    for (int m=0; m<nmb; ++m) {
      if (zp >= size.d_view(m).x3min && zp < size.d_view(m).x3max) {++ncand;}
    }
    int pick = static_cast<int>(SettleHash01(key)*static_cast<Real>(ncand));
    if (pick > ncand-1) {pick = ncand-1;}
    int m = 0, seen = 0;
    for (int mm=0; mm<nmb; ++mm) {
      if (zp >= size.d_view(mm).x3min && zp < size.d_view(mm).x3max) {
        if (seen == pick) {m = mm;}
        ++seen;
      }
    }
    Real ux = SettleHash01(key + UINT64_C(0x632be59bd9b4e019));
    Real uy = SettleHash01(key + UINT64_C(0x8cb92baa3f3d8dd7));
    int s = p % nspec;
    Real tau = taus_.d_view(s);
    Real d = 1.0 + kap*SQR(omega0*tau);
    pr(IPX,p) = size.d_view(m).x1min + ux*(size.d_view(m).x1max - size.d_view(m).x1min);
    pr(IPY,p) = size.d_view(m).x2min + uy*(size.d_view(m).x2max - size.d_view(m).x2min);
    pr(IPZ,p) = zp;
    pi(PGID,p) = gids + m;
    pi(PSP,p) = s;
    pr(IPTS,p) = tau;
    pr(IPM,p) = 1.0;
    pr(IPVX,p) = -2.0*omega0*tau*etavk/d;
    pr(IPVY,p) = -etavk/d;
    pr(IPVZ,p) = 0.0;
    pr(IPRX,p) = 0.0; pr(IPRY,p) = 0.0; pr(IPRZ,p) = 0.0;
  });
  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void DustSettleHistory()

void DustSettleHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  const int nspec = pmbp->pdust->nspecies;
  pdata->nhist = 5*nspec;
  for (int s=0; s<nspec; ++s) {
    std::string tag = std::to_string(s+1);
    pdata->label[5*s+0] = "zin_" + tag;
    pdata->label[5*s+1] = "nin_" + tag;
    pdata->label[5*s+2] = "zout_" + tag;
    pdata->label[5*s+3] = "nout_" + tag;
    pdata->label[5*s+4] = "zex_" + tag;
  }
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  const Real zc = 0.5*(pm->mesh_size.x3min + pm->mesh_size.x3max);
  const Real xsplit = settle_xsplit;
  for (int s=0; s<nspec; ++s) {
    Real zin = 0.0, nin = 0.0, zout = 0.0, nout = 0.0;
    Kokkos::parallel_reduce("settle_hist",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &zi, Real &ni, Real &zo, Real &no) {
      if (pi(PSP,p) != s) return;
      const bool in = (xsplit <= 0.0) || (fabs(pr(IPX,p)) < xsplit);
      if (in) {zi += pr(IPZ,p) - zc; ni += 1.0;} else {zo += pr(IPZ,p) - zc; no += 1.0;}
    }, Kokkos::Sum<Real>(zin), Kokkos::Sum<Real>(nin), Kokkos::Sum<Real>(zout),
       Kokkos::Sum<Real>(nout));
    pdata->hdata[5*s+0] = zin;
    pdata->hdata[5*s+1] = nin;
    pdata->hdata[5*s+2] = zout;
    pdata->hdata[5*s+3] = nout;
    // the analytic reference is the same on every rank; the history sums over ranks
    pdata->hdata[5*s+4] = SettleExact(pm->time, pmbp->pdust->taus.h_view(s),
                                      settle_omega, settle_z0)
                          /static_cast<Real>(global_variable::nranks);
  }
  return;
}

}  // namespace
