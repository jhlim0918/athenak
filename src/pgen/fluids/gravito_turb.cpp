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
#include <cstdint>
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

  // two-stage dust runs (Baehr, Zhu & Yang 2022 style): a gas-only run to saturation,
  // then a restart with <particles>/<dust> blocks and restart_insert = true, which
  // inserts the particles into the saturated state here
  if (restart) {
    if (pmbp->ppart != nullptr &&
        pin->GetOrAddBoolean("particles", "restart_insert", false)) {
      GravitoTurbInsertDust(pin);
    }
    return;
  }

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
//! \fn void ProblemGenerator::GravitoTurbInsertDust()
//! \brief Inserts the dust particles into a restarted (saturated) gravito-turbulent gas
//! state: <particles>/ppc particles per cell in total, spread uniformly in (x,y) and as
//! a Gaussian of width <problem>/dust_hz in z (Baehr et al. 2022: the initial gas
//! width), at rest in the shearing frame.  The <dust>/nspecies species (stopping times
//! <dust>/taus_s) are interleaved round-robin within every block, so each gets the same
//! number of particles and the same spatial distribution; species s carries the dust-to-
//! gas ratio <problem>/dust_Z_s (default: <problem>/dust_Z split evenly), its particles
//! having equal masses summing to dust_Z_s times the gas mass in the box.  Per-block
//! counts (rounded to a multiple of nspecies) follow the Gaussian mass in each block's
//! z range, so the placement is deterministic and decomposition-independent (hash of the
//! global block id and the particle's index in the block, <problem>/dust_seed).

void ProblemGenerator::GravitoTurbInsertDust(ParameterInput *pin) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "restart_insert needs a <dust> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->nprtcl_total != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "restart_insert on a restart file that already carries particles; set "
              << "particles/restart_insert = false to continue such a run" << std::endl;
    exit(EXIT_FAILURE);
  }
  particles::Particles *ppar = pmbp->ppart;
  auto &msize = pmy_mesh_->mesh_size;
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  Real ppc = pin->GetOrAddReal("particles", "ppc", 1.0);
  int nsp = pmbp->pdust->nspecies;
  Real dust_Z = pin->GetOrAddReal("problem", "dust_Z", 0.01);
  std::vector<Real> zs(nsp);
  for (int s=0; s<nsp; ++s) {
    zs[s] = pin->GetOrAddReal("problem", "dust_Z_" + std::to_string(s+1), dust_Z/nsp);
  }
  Real cs0 = pin->GetOrAddReal("problem", "cs0", 2.12625);
  Real dust_hz = pin->GetOrAddReal("problem", "dust_hz", cs0/gt_var.omega0);
  int64_t dust_seed = pin->GetOrAddInteger("problem", "dust_seed", 7);
  Real ncells_tot = static_cast<Real>(pmy_mesh_->mesh_indcs.nx1)
                   *static_cast<Real>(pmy_mesh_->mesh_indcs.nx2)
                   *static_cast<Real>(pmy_mesh_->mesh_indcs.nx3);
  Real n_target = ppc*ncells_tot;

  // gas mass in the box (active cells)
  auto &u0 = pmbp->phydro->u0;
  int ni = ie-is+1, nj = je-js+1, nk = ke-ks+1;
  int nmkji = nmb*nk*nj*ni;
  Real mgas = 0.0;
  Kokkos::parallel_reduce("gt_dust_mgas", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(int idx, Real &lsum) {
    int i = is + (idx % ni);
    int j = js + ((idx/ni) % nj);
    int k = ks + ((idx/(ni*nj)) % nk);
    int m = idx/(ni*nj*nk);
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    lsum += u0(m,IDN,k,j,i)*vol;
  }, Kokkos::Sum<Real>(mgas));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &mgas, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // radial pressure-gradient mimic: a constant a_x on the gas (<hydro_srcterms>/
  // const_accel, dir 1) is balanced by the uniform azimuthal offset v_y = -a_x/(2 Omega)
  // (the NSH gas velocity).  Apply it to the restarted gas here, so switching the force
  // on at stage 2 does not launch the undamped box-wide epicycle of amplitude eta v_K.
  // <problem>/gas_vy_shift = auto (default) or a number (0 = no shift).
  Real vy_auto = 0.0;
  auto *psrc = pmbp->phydro->psrc;
  if (psrc != nullptr && psrc->const_accel && psrc->const_accel_dir == 1) {
    vy_auto = -psrc->const_accel_val/(2.0*gt_var.omega0);
  }
  std::string vy_str = pin->GetOrAddString("problem", "gas_vy_shift", "auto");
  Real vy_shift = (vy_str.compare("auto") == 0) ? vy_auto : std::stod(vy_str);
  if (vy_shift != 0.0) {
    int ng = indcs.ng;
    bool is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
    par_for("gt_dust_vyshift", DevExeSpace(), 0, nmb-1, 0, indcs.nx3+2*ng-1,
            0, indcs.nx2+2*ng-1, 0, indcs.nx1+2*ng-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real den = u0(m,IDN,k,j,i);
      Real my = u0(m,IM2,k,j,i);
      u0(m,IM2,k,j,i) = my + den*vy_shift;
      if (is_ideal) u0(m,IEN,k,j,i) += my*vy_shift + 0.5*den*vy_shift*vy_shift;
    });
    if (global_variable::my_rank == 0) {
      std::cout << "gravito_turb: gas azimuthal velocity shifted by " << vy_shift
                << " (pressure-gradient equilibrium)" << std::endl;
    }
  }

  // per-block counts: (x,y) area fraction times the Gaussian mass in the block's z range
  Real area_box = gt_var.lx*gt_var.ly;
  Real s2 = dust_hz*std::sqrt(2.0);
  Real znorm = 0.5*(std::erf(msize.x3max/s2) - std::erf(msize.x3min/s2));
  std::vector<int> off(nmb + 1, 0);
  for (int m=0; m<nmb; ++m) {
    Real area = (size.h_view(m).x1max - size.h_view(m).x1min)
               *(size.h_view(m).x2max - size.h_view(m).x2min);
    Real zfrac = 0.5*(std::erf(size.h_view(m).x3max/s2)
                      - std::erf(size.h_view(m).x3min/s2))/znorm;
    int n_m = nsp*static_cast<int>(std::floor(n_target*(area/area_box)*zfrac/nsp + 0.5));
    off[m+1] = off[m] + n_m;
  }
  int npart = off[nmb];
  int64_t ntot = npart;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &ntot, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
  // ntot/nsp particles per species; species s has mass zs[s]*mgas in total
  DualArray1D<Real> mps("gt_dust_mp", nsp);
  for (int s=0; s<nsp; ++s) {
    mps.h_view(s) = zs[s]*mgas*static_cast<Real>(nsp)
                    /static_cast<Real>(std::max<int64_t>(ntot, 1));
  }
  mps.template modify<HostMemSpace>();
  mps.template sync<DevExeSpace>();

  ppar->nprtcl_thispack = npart;
  Kokkos::realloc(ppar->prtcl_rdata, ppar->nrdata, std::max(npart, 1));
  Kokkos::realloc(ppar->prtcl_idata, ppar->nidata, std::max(npart, 1));
  DvceArray1D<int> d_off("gt_dust_off", nmb + 1);
  {
    auto h_off = Kokkos::create_mirror_view(d_off);
    for (int m=0; m<=nmb; ++m) h_off(m) = off[m];
    Kokkos::deep_copy(d_off, h_off);
  }
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto gids = pmbp->gids;
  auto &taus_ = pmbp->pdust->taus;
  Real hz = dust_hz;
  int64_t seed = dust_seed;
  par_for("gt_dust_insert", DevExeSpace(), 0, (npart-1), KOKKOS_LAMBDA(const int p) {
    // owning block by binary search over the offsets
    int lo = 0, hi = nmb - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1)/2;
      if (d_off(mid) <= p) {lo = mid;} else {hi = mid - 1;}
    }
    int m = lo;
    int64_t q = p - d_off(m);
    int64_t gid = gids + m;
    Real ux = 0.5*(HashNoise(gid, q, 0, 0, seed) + 1.0);
    Real uy = 0.5*(HashNoise(gid, q, 1, 0, seed) + 1.0);
    Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;
    pr(IPX,p) = x1min + ux*(x1max - x1min);
    pr(IPY,p) = x2min + uy*(x2max - x2min);
    // Gaussian z truncated to the block: Box-Muller from hashed uniforms, rejection
    Real z = 0.5*(x3min + x3max);
    for (int trial=0; trial<64; ++trial) {
      Real u1 = 0.5*(HashNoise(gid, q, 2, trial, seed) + 1.0);
      Real u2 = 0.5*(HashNoise(gid, q, 3, trial, seed) + 1.0);
      u1 = fmax(u1, 1.0e-300);
      Real zt = hz*sqrt(-2.0*log(u1))*cos(6.283185307179586*u2);
      if (zt >= x3min && zt < x3max) {z = zt; break;}
    }
    pr(IPZ,p) = z;
    int sp = static_cast<int>(q % nsp);
    pi(PGID,p) = static_cast<int>(gid);
    pi(PSP,p) = sp;
    pr(IPTS,p) = taus_.d_view(sp);
    pr(IPM,p) = mps.d_view(sp);
    pr(IPVX,p) = 0.0; pr(IPVY,p) = 0.0; pr(IPVZ,p) = 0.0;
    pr(IPRX,p) = 0.0; pr(IPRY,p) = 0.0; pr(IPRZ,p) = 0.0;
  });

  // Mesh bookkeeping and tags
  Mesh *pm = pmy_mesh_;
  pm->nprtcl_thisrank = npart;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&(pm->nprtcl_thisrank), 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT,
                MPI_COMM_WORLD);
#else
  pm->nprtcl_eachrank[0] = npart;
#endif
  pm->nprtcl_total = 0;
  for (int r=0; r<global_variable::nranks; ++r) {
    pm->nprtcl_total += pm->nprtcl_eachrank[r];
  }
  ppar->CreateParticleTags(pin);
  if (global_variable::my_rank == 0) {
    std::cout << "gravito_turb: inserted " << pm->nprtcl_total << " dust particles "
              << "(target " << static_cast<int64_t>(n_target) << ") in " << nsp
              << " species, gas mass " << mgas << ", Gaussian h_z = " << dust_hz
              << std::endl;
    for (int s=0; s<nsp; ++s) {
      std::cout << "gravito_turb:   species " << s+1 << ": stopping time "
                << taus_.h_view(s) << ", Z = " << zs[s] << ", "
                << pm->nprtcl_total/nsp << " particles of mass " << mps.h_view(s)
                << std::endl;
    }
  }
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
