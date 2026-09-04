//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gravito_turb.cpp
//! \brief 3D stratified, self-gravitating, cooling shearing box: the gravito-turbulence
//! setup of Shi & Chiang (2014), ApJ 789, 34 (SC14).
//!
//! Initial condition (SC14 sec. 2.3): vertical hydrostatic equilibrium of a polytrope
//! P = K*rho^gamma under linearized vertical tidal gravity and its own self-gravity,
//!     (1/rho) dP/dz = -Omega^2 z - 4 pi G int_0^z rho dz',
//! integrated from the midplane (rho0 = 1) with K = cs0^2/(gamma*rho0^(gamma-1)).
//! SC14's fiducial constants (code units rho0 = H = Omega = 1, Toomre Q0 = 1,
//! gamma = 5/3): cs0 = 2.126, K = 2.712, G = 0.3384 (four_pi_G = 4.2523).  Where the
//! profile falls below dfloor_ic the density is set to that floor (cold halo).
//! Random cell-to-cell velocity perturbations up to pert_amp*cs0 are seeded for
//! |z| < pert_zmax (deterministic integer-hash RNG: rank- and decomposition-
//! independent).
//!
//! Physics configuration expected from the input file:
//!   <shearing_box> qshear = 1.5, omega0 = 1, stratified = true
//!   <hydro_srcterms> self_gravity = true, beta_cooling = true (bcool_beta = beta)
//!   <gravity> solver = multigrid, mg_bc = slab  (or solver = fft, vert_bc = open)
//!   <mesh> ix3_bc/ox3_bc = outflow;  <hydro> dfloor for the evolution floor
//!
//! User history (volume integrals; ratios formed in post-processing):
//!   0: mass       total mass in box
//!   1: rho_cs     int rho*cs dV            (-> <cs>_rho = h1/h0; Q = h1/h0*Omega/
//!                                              (pi G h0/(Lx*Ly)), SC14 eq. 16)
//!   2: rho_wgrv   int rho*(gx*gy/4piG) dV   \  density-weighted stresses:
//!   3: rho_wrey   int rho*(rho*vx*dvy) dV   /  alpha = (h2+h3)/h4  (SC14 eq. 19)
//!   4: rho_prs    int rho*P dV
//!   5: wgrv       int gx*gy/4piG dV         \  volume-weighted stresses:
//!   6: wrey       int rho*vx*dvy dV         /  alpha' = (2/3)(h5+h6)/h7 (eq. 20;
//!                                              h7 = gamma*int P absorbs the 1/gamma)
//!   7: rho_cs2    int rho*cs^2 dV
//!   8: rho_dv2    int rho*dv^2 dV           (-> rms dv = sqrt(h8/h0), dv includes
//!                                              vx, dvy, vz;  SC14 eq. 18)
//!   9: eint       int P/(gamma-1) dV        (thermal energy; cooling balance checks)

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "shearing_box/shearing_box.hpp"
#include "srcterms/srcterms.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "pgen/pgen.hpp"

// User-defined history function
void GravitoTurbHistory(HistoryData *pdata, Mesh *pm);

namespace {
struct GravitoTurbVariables {
  Real qshear, omega0, four_pi_G, gamma;
  Real lx, ly;
  bool orbital_advection;
};
GravitoTurbVariables gt_var;

// deterministic integer hash -> uniform Real in [-1, 1): splitmix64 finalizer
KOKKOS_INLINE_FUNCTION
Real HashNoise(int64_t gk, int64_t gj, int64_t gi, int64_t c, int64_t seed) {
  uint64_t x = static_cast<uint64_t>(seed);
  x += 0x9e3779b97f4a7c15ULL*static_cast<uint64_t>(gk + 1);
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x += 0x9e3779b97f4a7c15ULL*static_cast<uint64_t>(gj + 1);
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  x += 0x9e3779b97f4a7c15ULL*static_cast<uint64_t>(gi + 1);
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x += 0x9e3779b97f4a7c15ULL*static_cast<uint64_t>(c + 1);
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return 2.0*(static_cast<Real>(x >> 11)/9007199254740992.0) - 1.0;  // 53-bit mantissa
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::GravitoTurb()
//! \brief sets up the SC14 stratified self-gravitating cooling shearing box

void ProblemGenerator::GravitoTurb(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "gravito_turb is hydro-only" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->phydro->psbox_u == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "gravito_turb requires a <shearing_box> block in the input file"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmbp->phydro->psbox_u->is_stratified) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "gravito_turb requires <shearing_box> stratified = true" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pgrav == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "gravito_turb requires a <gravity> block in the input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // gravitational constant (same duplicated-setter pattern as the other gravity pgens)
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 4.25231);
  pmbp->pgrav->four_pi_G = four_pi_G;
  if (pmbp->pgrav->pmgd != nullptr) {
    pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);
  }

  auto &msize = pmy_mesh_->mesh_size;
  gt_var.qshear = pmbp->phydro->psbox_u->qshear;
  gt_var.omega0 = pmbp->phydro->psbox_u->omega0;
  gt_var.four_pi_G = four_pi_G;
  gt_var.gamma = pmbp->phydro->peos->eos_data.gamma;
  gt_var.lx = msize.x1max - msize.x1min;
  gt_var.ly = msize.x2max - msize.x2min;
  gt_var.orbital_advection = pmbp->phydro->psbox_u->orbital_advection;
  user_hist_func = GravitoTurbHistory;

  if (restart) return;

  // problem parameters (defaults = SC14 fiducial constants)
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real cs0 = pin->GetOrAddReal("problem", "cs0", 2.12625);
  Real dfloor_ic = pin->GetOrAddReal("problem", "dfloor_ic", 1.0e-4);
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.1);
  Real pert_zmax = pin->GetOrAddReal("problem", "pert_zmax", 2.0);
  int pert_seed = pin->GetOrAddInteger("problem", "pert_seed", 1);

  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gamma = eos.gamma;
  Real gm1 = gamma - 1.0;
  if (!eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "gravito_turb requires an ideal-gas EOS" << std::endl;
    exit(EXIT_FAILURE);
  }
  Real omega0 = gt_var.omega0;
  Real kpoly = cs0*cs0/(gamma*std::pow(rho0, gm1));

  // ---- integrate the vertical hydrostatic equilibrium (SC14 eq. 9) -------------------
  // d(rho)/dz = -rho*(Omega^2 z + 4 pi G mcol) / (gamma*K*rho^(gamma-1)),
  // d(mcol)/dz = rho, from rho(0) = rho0 on a fine table; RK2 (midpoint) is ample.
  Real zmax = std::max(std::abs(msize.x3min), std::abs(msize.x3max));
  const int nfine = 1 << 15;
  Real hfine = zmax/static_cast<Real>(nfine);
  std::vector<Real> rho_tab(nfine + 1);
  {
    Real rho = rho0, mcol = 0.0;
    Real rfloor = dfloor_ic*rho0;
    rho_tab[0] = rho;
    auto drho = [&](Real z, Real r, Real mc) {
      return -r*(SQR(omega0)*z + four_pi_G*mc)/(gamma*kpoly*std::pow(r, gm1));
    };
    for (int l = 0; l < nfine; ++l) {
      Real z = l*hfine;
      if (rho <= rfloor) {
        rho_tab[l+1] = rfloor;
        continue;
      }
      // midpoint step
      Real k1r = drho(z, rho, mcol);
      Real k1m = rho;
      Real rmid = std::max(rho + 0.5*hfine*k1r, rfloor);
      Real mmid = mcol + 0.5*hfine*k1m;
      Real k2r = drho(z + 0.5*hfine, rmid, mmid);
      Real k2m = rmid;
      rho = std::max(rho + hfine*k2r, rfloor);
      mcol = mcol + hfine*k2m;
      rho_tab[l+1] = rho;
    }
  }
  DvceArray1D<Real> d_rho_tab("gt_rhotab", nfine + 1);
  {
    auto h_tab = Kokkos::create_mirror_view(d_rho_tab);
    for (int l = 0; l <= nfine; ++l) h_tab(l) = rho_tab[l];
    Kokkos::deep_copy(d_rho_tab, h_tab);
  }
  if (global_variable::my_rank == 0) {
    // effective half-thickness h_eff = (1/2) int rho/rho0 dz (SC14 eq. 11 analogue in
    // code units; SC14's H = h*H_sg = 1 by construction for the fiducial constants)
    Real halfint = 0.0;
    for (int l = 0; l < nfine; ++l) {
      halfint += 0.5*(rho_tab[l] + rho_tab[l+1])*hfine;
    }
    std::cout << "gravito_turb IC: cs0 = " << cs0 << ", K = " << kpoly
              << ", four_pi_G = " << four_pi_G << std::endl
              << "gravito_turb IC: half-thickness (1/rho0) int_0^zmax rho dz = "
              << halfint/rho0 << " (SC14 fiducial: ~1)" << std::endl;
  }

  // ---- fill the grid -----------------------------------------------------------------
  bool orb_adv = gt_var.orbital_advection;
  Real qsh = gt_var.qshear;
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  Real dx1m = gt_var.lx/static_cast<Real>(pmy_mesh_->mesh_indcs.nx1);
  Real dx2m = gt_var.ly/static_cast<Real>(pmy_mesh_->mesh_indcs.nx2);
  Real dx3m = (msize.x3max - msize.x3min)/static_cast<Real>(pmy_mesh_->mesh_indcs.nx3);
  Real x1min_m = msize.x1min, x2min_m = msize.x2min, x3min_m = msize.x3min;
  Real vpert = pert_amp*cs0;
  int64_t seed = pert_seed;

  par_for("gt_init", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // interpolate the equilibrium table at |z|
    Real az = fabs(x3v);
    Real rl = az/hfine;
    int l = static_cast<int>(rl);
    l = (l > nfine-1) ? nfine-1 : l;
    Real wgt = rl - static_cast<Real>(l);
    Real den = (1.0-wgt)*d_rho_tab(l) + wgt*d_rho_tab(l+1);
    Real prs = kpoly*pow(den, gamma);

    // deterministic random velocity perturbations for |z| < pert_zmax (global
    // root-grid cell indices make this decomposition-independent)
    Real vx = 0.0, vy = 0.0, vz = 0.0;
    if (az < pert_zmax) {
      int64_t gi = static_cast<int64_t>((x1v - x1min_m)/dx1m);
      int64_t gj = static_cast<int64_t>((x2v - x2min_m)/dx2m);
      int64_t gk = static_cast<int64_t>((x3v - x3min_m)/dx3m);
      vx = vpert*HashNoise(gk, gj, gi, 0, seed);
      vy = vpert*HashNoise(gk, gj, gi, 1, seed);
      vz = vpert*HashNoise(gk, gj, gi, 2, seed);
    }
    // non-FARGO frame carries the background shear flow
    if (!orb_adv) vy += -qsh*omega0*x1v;

    u0(m,IDN,k,j,i) = den;
    u0(m,IM1,k,j,i) = den*vx;
    u0(m,IM2,k,j,i) = den*vy;
    u0(m,IM3,k,j,i) = den*vz;
    u0(m,IEN,k,j,i) = prs/gm1 + 0.5*den*(SQR(vx) + SQR(vy) + SQR(vz));
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn GravitoTurbHistory()
//! \brief SC14 diagnostics: mass, Toomre-Q ingredients, density- and volume-weighted
//! stresses (eqs. 19-20), rms velocity fluctuations, thermal energy

void GravitoTurbHistory(HistoryData *pdata, Mesh *pm) {
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;
  auto gv = gt_var;

  pdata->nhist = 10;
  pdata->label[0] = "mass";
  pdata->label[1] = "rho_cs";
  pdata->label[2] = "rho_wgrv";
  pdata->label[3] = "rho_wrey";
  pdata->label[4] = "rho_prs";
  pdata->label[5] = "wgrv";
  pdata->label[6] = "wrey";
  pdata->label[7] = "rho_cs2";
  pdata->label[8] = "rho_dv2";
  pdata->label[9] = "eint";
  // dust track (Phase 4d): the particles' mass, momenta (shear-relative), kinetic energy,
  // Reynolds stress and the mass removed through the vertical faces
  dust::DustGasDrag *pdust = pm->pmb_pack->pdust;
  const bool has_dust = (pdust != nullptr);
  if (has_dust) {
    pdata->nhist = 17;
    pdata->label[10] = "d_mass";
    pdata->label[11] = "d_px";
    pdata->label[12] = "d_py";
    pdata->label[13] = "d_pz";
    pdata->label[14] = "d_ke";
    pdata->label[15] = "d_wrey";
    pdata->label[16] = "d_escaped";
  }

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &w0_ = pm->pmb_pack->phydro->w0;
  auto &phi_ = pm->pmb_pack->pgrav->phi;
  Real gamma = gv.gamma;
  Real gm1 = gamma - 1.0;
  Real ifpg = 1.0/gv.four_pi_G;
  Real qom = gv.qshear*gv.omega0;
  bool orb_adv = gv.orbital_advection;
  array_sum::GlobalSum sum_this_mb;

  Kokkos::parallel_reduce("GTurbHist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;
    Real vol = dx1*dx2*dx3;
    Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real den = w0_(m,IDN,k,j,i);
    Real prs = gm1*w0_(m,IEN,k,j,i);
    Real cs = sqrt(gamma*prs/den);
    Real vx = w0_(m,IVX,k,j,i);
    Real vz = w0_(m,IVZ,k,j,i);
    // non-Keplerian azimuthal velocity: the FARGO frame carries it directly
    Real dvy = orb_adv ? w0_(m,IVY,k,j,i) : w0_(m,IVY,k,j,i) + qom*x1v;

    // local self-gravitational acceleration from the potential
    Real gx = -(phi_(m,0,k,j,i+1) - phi_(m,0,k,j,i-1))/(2.0*dx1);
    Real gy = -(phi_(m,0,k,j+1,i) - phi_(m,0,k,j-1,i))/(2.0*dx2);

    Real wgrv = gx*gy*ifpg;
    Real wrey = den*vx*dvy;
    Real dv2 = SQR(vx) + SQR(dvy) + SQR(vz);

    array_sum::GlobalSum hvars;
    hvars.the_array[0] = vol*den;
    hvars.the_array[1] = vol*den*cs;
    hvars.the_array[2] = vol*den*wgrv;
    hvars.the_array[3] = vol*den*wrey;
    hvars.the_array[4] = vol*den*prs;
    hvars.the_array[5] = vol*wgrv;
    hvars.the_array[6] = vol*wrey;
    hvars.the_array[7] = vol*den*cs*cs;
    hvars.the_array[8] = vol*den*dv2;
    hvars.the_array[9] = vol*prs/gm1;
    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
  Kokkos::fence();

  for (int n=0; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }
  if (has_dust) {
    particles::Particles *ppar = pm->pmb_pack->ppart;
    auto &pr = ppar->prtcl_rdata;
    int npart = ppar->nprtcl_thispack;
    Real dm = 0.0, dpx = 0.0, dpy = 0.0, dpz = 0.0, dke = 0.0, dwr = 0.0;
    Kokkos::parallel_reduce("GTurbDustHist",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int &p, Real &m_, Real &px_, Real &py_, Real &pz_, Real &ke_,
                  Real &wr_) {
      Real mp = pr(IPM,p);
      Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);
      m_ += mp;
      px_ += mp*vx;
      py_ += mp*vy;
      pz_ += mp*vz;
      ke_ += 0.5*mp*(vx*vx + vy*vy + vz*vz);
      wr_ += mp*vx*vy;
    }, Kokkos::Sum<Real>(dm), Kokkos::Sum<Real>(dpx), Kokkos::Sum<Real>(dpy),
       Kokkos::Sum<Real>(dpz), Kokkos::Sum<Real>(dke), Kokkos::Sum<Real>(dwr));
    pdata->hdata[10] = dm;
    pdata->hdata[11] = dpx;
    pdata->hdata[12] = dpy;
    pdata->hdata[13] = dpz;
    pdata->hdata[14] = dke;
    pdata->hdata[15] = dwr;
    pdata->hdata[16] = pdust->escaped_mass;
  }
  return;
}
