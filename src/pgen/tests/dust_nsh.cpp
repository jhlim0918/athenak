//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_nsh.cpp
//! \brief Problem generator for the multi-species Nakagawa-Sei-Hayashi (NSH) drift
//! equilibrium test in an unstratified shearing box (Nakagawa et al. 1986;
//! Benitez-Llambay et al. 2019 Sec. 3.4; Krapp et al. 2024 Sec. 3.6). The gas feels a
//! constant radial acceleration a_x = 2*Omega*etavk (the pressure-gradient mimic,
//! applied through the standard <hydro_srcterms> const_accel machinery so that dust
//! particles do not feel it), and gas + N dust species settle into uniform steady drift
//! velocities that solve the linear system
//!     0 =  2*Omega*u_phi + a_x + sum_s eps_s*(v_x,s - u_x)/tau_s
//!     0 = -(2-q)*Omega*u_x     + sum_s eps_s*(v_phi,s - u_phi)/tau_s
//!     0 =  2*Omega*v_phi,s - (v_x,s - u_x)/tau_s
//!     0 = -(2-q)*Omega*v_x,s - (v_phi,s - u_phi)/tau_s
//! (velocities relative to the background shear). By default the problem uses a quiet
//! particle lattice, initializes this equilibrium exactly, and verifies it is HELD TO
//! ROUND-OFF: because the
//! rotation/shear/forcing kicks and the implicit drag solve are composed unsplit inside
//! the IMEX stages, the discrete update has the continuum equilibrium as an exact fixed
//! point (any Strang-like splitting error would appear as secular drift).
//! <problem>/particle_placement=random instead gives a reproducible warm start at the
//! same NSH velocities for nonlinear streaming-instability calculations.
//! A physical dust-free equilibrium is obtained by omitting both <particles> and <dust>.
//! In that mode there are no dust species or stopping times, so the solution reduces to
//! u_x=0 and u_phi=-etavk.
//!
//! Works in the 2D r-z shearing box (azimuthal components in IM3/IPVZ, plain periodic
//! boundaries) and in 3D (azimuthal in IM2/IPVY, shear-periodic x1 boundaries).
//! Errors written to "<basename>-errs.dat"; per-species drift history available with
//! <problem>/user_hist = true.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>     // fopen(), fprintf()
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "outputs/outputs.hpp"

// functions for error output and user history
void DustNSHErrors(ParameterInput *pin, Mesh *pm);
void DustNSHHistory(HistoryData *pdata, Mesh *pm);

namespace {

//----------------------------------------------------------------------------------------
//! \fn HashUniform01
//! \brief Reproducible stateless pseudo-random number in [0,1), suitable for device code.

KOKKOS_INLINE_FUNCTION
Real HashUniform01(std::uint64_t key) {
  // SplitMix64 finalizer. Using particle tags as keys makes the initial condition
  // independent of Kokkos execution order and therefore reproducible on CPU and GPU.
  key += UINT64_C(0x9e3779b97f4a7c15);
  key = (key ^ (key >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  key = (key ^ (key >> 27)) * UINT64_C(0x94d049bb133111eb);
  key ^= key >> 31;
#if SINGLE_PRECISION_ENABLED
  // All integers through 2^24 are exactly representable as float.  Using 24 bits keeps
  // the largest result at 1 - 2^-24; casting a 53-bit value to float could round to 1.
  return static_cast<Real>(key >> 40) * static_cast<Real>(1.0/16777216.0);
#else
  return static_cast<Real>(key >> 11) * static_cast<Real>(1.0/9007199254740992.0);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn SolveNSH
//! \brief Solves the (2+2N)x(2+2N) linear system for the multi-species NSH drift
//! equilibrium by Gaussian elimination with partial pivoting. Unknown ordering:
//! [u_x, u_phi, v_x,1, v_phi,1, ..., v_x,N, v_phi,N].

void SolveNSH(const Real omega0, const Real qshear, const Real ax,
              const std::vector<Real> &eps, const std::vector<Real> &taus,
              Real &ugx, Real &ugp, std::vector<Real> &vx, std::vector<Real> &vp) {
  int ns = static_cast<int>(eps.size());
  if (ns == 0) {
    ugx = 0.0;
    ugp = -ax/(2.0*omega0);
    vx.clear();
    vp.clear();
    return;
  }
  int n = 2 + 2*ns;
  std::vector<std::vector<Real>> a(n, std::vector<Real>(n+1, 0.0));

  // gas radial:  2*Om*u_phi + sum eps_s/tau_s*(v_x,s - u_x) = -a_x
  a[0][1] = 2.0*omega0;
  for (int s=0; s<ns; ++s) {
    a[0][0]     -= eps[s]/taus[s];
    a[0][2+2*s] += eps[s]/taus[s];
  }
  a[0][n] = -ax;
  // gas azimuthal: -(2-q)*Om*u_x + sum eps_s/tau_s*(v_phi,s - u_phi) = 0
  a[1][0] = -(2.0-qshear)*omega0;
  for (int s=0; s<ns; ++s) {
    a[1][1]     -= eps[s]/taus[s];
    a[1][3+2*s] += eps[s]/taus[s];
  }
  // dust radial / azimuthal for each species
  for (int s=0; s<ns; ++s) {
    int rx = 2+2*s, rp = 3+2*s;
    a[rx][rp] = 2.0*omega0;
    a[rx][rx] -= 1.0/taus[s];
    a[rx][0]  += 1.0/taus[s];
    a[rp][rx] = -(2.0-qshear)*omega0;
    a[rp][rp] -= 1.0/taus[s];
    a[rp][1]  += 1.0/taus[s];
  }

  // Gaussian elimination with partial pivoting
  for (int c=0; c<n; ++c) {
    int piv = c;
    for (int r=c+1; r<n; ++r) {
      if (fabs(a[r][c]) > fabs(a[piv][c])) {piv = r;}
    }
    std::swap(a[c], a[piv]);
    for (int r=c+1; r<n; ++r) {
      Real f = a[r][c]/a[c][c];
      for (int cc=c; cc<=n; ++cc) {a[r][cc] -= f*a[c][cc];}
    }
  }
  std::vector<Real> x(n);
  for (int r=n-1; r>=0; --r) {
    Real sum = a[r][n];
    for (int cc=r+1; cc<n; ++cc) {sum -= a[r][cc]*x[cc];}
    x[r] = sum/a[r][r];
  }

  ugx = x[0];
  ugp = x[1];
  vx.resize(ns);
  vp.resize(ns);
  for (int s=0; s<ns; ++s) {
    vx[s] = x[2+2*s];
    vp[s] = x[3+2*s];
  }
}

//----------------------------------------------------------------------------------------
//! \fn ReadNSHParams
//! \brief Reads the shearing-box and active dust-mixture parameters.

void ReadNSHParams(ParameterInput *pin, MeshBlockPack *pmbp,
                   Real &omega0, Real &qshear, Real &etavk,
                   std::vector<Real> &eps, std::vector<Real> &taus) {
  dust::DustGasDrag *pdust = pmbp->pdust;
  int ns = (pdust != nullptr) ? pdust->nspecies : 0;
  omega0 = pin->GetReal("shearing_box","omega0");
  qshear = pin->GetReal("shearing_box","qshear");
  etavk  = pin->GetReal("problem","etavk");
  eps.resize(ns);
  taus.resize(ns);
  bool has_eps = pin->DoesParameterExist("problem","eps_1");
  for (int s=0; s<ns; ++s) {
    if (has_eps) {
      eps[s] = pin->GetReal("problem","eps_" + std::to_string(s+1));
    } else {
      eps[s] = pdust->dust_to_gas/static_cast<Real>(ns);
    }
    taus[s] = pdust->taus.h_view(s);
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \brief AMR test schedule (<problem>/amr_test = true): MeshBlocks intersecting the box
//! [amr_x1min, amr_x1max] x (all x2) x [amr_x3min, amr_x3max] are flagged for refinement
//! while amr_t_on <= t < amr_t_off and for derefinement otherwise, so the lattice
//! equilibrium passes through one refine and one derefine event.

namespace {
bool nsh_amr_test = false;
bool nsh_amr_density = false;
Real nsh_amr_x1min, nsh_amr_x1max, nsh_amr_x3min, nsh_amr_x3max;
Real nsh_amr_t_on, nsh_amr_t_off;
Real nsh_amr_hold, nsh_amr_rho_lev[8], nsh_amr_hyst;
int nsh_amr_nthr = 0;

void DustNSHRefine(MeshBlockPack *pmbp) {
  if (!nsh_amr_test && !nsh_amr_density) return;
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  int nmb = pmbp->nmb_thispack;
  Real t = pmbp->pmesh->time;
  if (nsh_amr_test) {
    auto &size = pmbp->pmb->mb_size;
    bool on = (t >= nsh_amr_t_on) && (t < nsh_amr_t_off);
    for (int m=0; m<nmb; ++m) {
      bool inx1 = (size.h_view(m).x1max > nsh_amr_x1min) &&
                  (size.h_view(m).x1min < nsh_amr_x1max);
      bool inx3 = (!pmbp->pmesh->three_d) ||
                  ((size.h_view(m).x3max > nsh_amr_x3min) &&
                   (size.h_view(m).x3min < nsh_amr_x3max));
      refine_flag.h_view(m + mbs) = (on && inx1 && inx3) ? 1 : -1;
    }
  } else {
    // density mode: before the hold time every block refines (the linear growth needs
    // the finest resolution everywhere); after it the target level of a block is the
    // number of thresholds amr_rho_lev1 < amr_rho_lev2 < ... its maximum PM dust density
    // exceeds, and the flag moves the block one level toward the target
    int root = pmbp->pmesh->root_level;
    if (t < nsh_amr_hold) {
      for (int m=0; m<nmb; ++m) {refine_flag.h_view(m + mbs) = 1;}
    } else {
      pmbp->pdust->AssembleDustDensityNow();   // collective
      auto &indcs = pmbp->pmesh->mb_indcs;
      int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
      int ks = indcs.ks, ke = indcs.ke;
      auto &rho = pmbp->pdust->rho_dust;
      DvceArray1D<Real> bmax("nsh_amr_bmax", nmb);
      par_for_outer("nsh_amr_max", DevExeSpace(), 0, 0, 0, nmb-1,
      KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
        Real mx = 0.0;
        int nkji = (ke-ks+1)*(je-js+1)*(ie-is+1);
        int nji = (je-js+1)*(ie-is+1);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &q) {
          int k = idx/nji;
          int j = (idx - k*nji)/(ie-is+1);
          int i = (idx - k*nji - j*(ie-is+1)) + is;
          q = fmax(q, rho(m,0,k+ks,j+js,i));
        }, Kokkos::Max<Real>(mx));
        bmax(m) = mx;
      });
      auto bmax_h = Kokkos::create_mirror_view(bmax);
      Kokkos::deep_copy(bmax_h, bmax);
      for (int m=0; m<nmb; ++m) {
        int target = 0;
        for (int n=0; n<nsh_amr_nthr; ++n) {
          if (bmax_h(m) > nsh_amr_rho_lev[n]) {target = n + 1;}
        }
        int lev = pmbp->pmb->mb_lev.h_view(m) - root;
        int flag = 0;
        if (target > lev) {
          flag = 1;
        } else if (target < lev) {
          // hysteresis: leave a level only once the maximum has fallen below a fraction
          // of that level's threshold (no flapping at the threshold)
          if ((lev > nsh_amr_nthr) ||
              (bmax_h(m) < nsh_amr_hyst*nsh_amr_rho_lev[lev-1])) {flag = -1;}
        }
        refine_flag.h_view(m + mbs) = flag;
      }
    }
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustNSH()
//! \brief Problem Generator for the multi-species NSH drift equilibrium

void ProblemGenerator::DustNSH(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool has_particles = (pmbp->ppart != nullptr);
  bool has_dust = (pmbp->pdust != nullptr);
  if (has_particles != has_dust) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "NSH requires both <particles> and <dust>, or neither "
              << "for the dust-free gas equilibrium" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool dust_active = has_particles && has_dust;
  bool random_placement = false;
  if (dust_active) {
    std::string placement =
        pin->GetOrAddString("problem","particle_placement","lattice");
    random_placement = (placement.compare("random") == 0);
    if (!random_placement && placement.compare("lattice") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<problem>/particle_placement must be lattice or random"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  bool check_equilibrium =
      pin->GetOrAddBoolean("problem","check_equilibrium",!random_placement);
  // SMR: place the lattice at the FINEST-level cell centres on every MeshBlock (blocks
  // one level below the finest carry 2^d times as many particles, with masses scaled to
  // the finest cell volume), so the finest-kernel deposits are exactly uniform across
  // the level boundaries and the equilibrium is again held to round-off
  bool lattice_finest = pin->GetOrAddBoolean("problem","lattice_finest",false);
  if (lattice_finest && (random_placement || !(pmy_mesh_->multilevel))) {
    lattice_finest = false;
  }
  pgen_final_func = check_equilibrium ? DustNSHErrors : nullptr;
  // Keep this enrolled in dust-free runs as well so an otherwise unchanged input
  // with user_hist=true remains valid; the callback emits no dust columns in that mode.
  user_hist_func = DustNSHHistory;
  // AMR test schedule (see DustNSHRefine)
  nsh_amr_test = pin->GetOrAddBoolean("problem","amr_test",false);
  if (nsh_amr_test) {
    nsh_amr_x1min = pin->GetReal("problem","amr_x1min");
    nsh_amr_x1max = pin->GetReal("problem","amr_x1max");
    nsh_amr_x3min = pin->GetOrAddReal("problem","amr_x3min",-1.0e300);
    nsh_amr_x3max = pin->GetOrAddReal("problem","amr_x3max", 1.0e300);
    nsh_amr_t_on  = pin->GetReal("problem","amr_t_on");
    nsh_amr_t_off = pin->GetReal("problem","amr_t_off");
    user_ref_func = DustNSHRefine;
  }
  // AMR density mode (see DustNSHRefine): amr_hold_time, amr_rho_lev1..N thresholds
  nsh_amr_density = pin->GetOrAddBoolean("problem","amr_density",false);
  if (nsh_amr_density) {
    nsh_amr_hold = pin->GetOrAddReal("problem","amr_hold_time",0.0);
    nsh_amr_hyst = pin->GetOrAddReal("problem","amr_hyst",0.5);
    nsh_amr_nthr = 0;
    for (int n=1; n<=8; ++n) {
      std::string key = "amr_rho_lev" + std::to_string(n);
      if (!pin->DoesParameterExist("problem", key)) break;
      nsh_amr_rho_lev[n-1] = pin->GetReal("problem", key);
      nsh_amr_nthr = n;
    }
    user_ref_func = DustNSHRefine;
  }
  if (restart) return;

  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH test requires a <hydro> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(pin->DoesBlockExist("shearing_box"))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH test requires a <shearing_box> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real omega0, qshear, etavk;
  std::vector<Real> eps, taus;
  ReadNSHParams(pin, pmbp, omega0, qshear, etavk, eps, taus);
  int nspec = static_cast<int>(eps.size());
  Real ax = 2.0*omega0*etavk;

  // the radial forcing must be applied to the gas by the standard const_accel source
  // term (so that dust particles do not feel it); verify the input file sets it up
  {
    bool ok = pin->GetOrAddBoolean("hydro_srcterms","const_accel",false);
    Real val = ok ? pin->GetReal("hydro_srcterms","const_accel_val") : 0.0;
    int dir = ok ? pin->GetInteger("hydro_srcterms","const_accel_dir") : 0;
    if (!ok || dir != 1 || fabs(val - ax) > 1.0e-12*fabs(ax)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "NSH test requires <hydro_srcterms> const_accel = true, "
                << "const_accel_dir = 1, const_accel_val = 2*omega0*etavk = " << ax
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // equilibrium drift velocities
  Real ugx, ugp;
  std::vector<Real> vx, vp;
  SolveNSH(omega0, qshear, ax, eps, taus, ugx, ugp, vx, vp);
  if (global_variable::my_rank == 0) {
    if (dust_active) {
      std::cout << "# NSH equilibrium: u_gx=" << ugx << " u_gphi=" << ugp << std::endl;
      for (int s=0; s<nspec; ++s) {
        std::cout << "#   species " << s+1 << ": v_x=" << vx[s]
                  << " v_phi=" << vp[s] << std::endl;
      }
    } else {
      std::cout << "# Dust-free NSH equilibrium: u_gx=" << ugx
                << " u_gphi=" << ugp << std::endl;
    }
  }

  // initialize uniform gas at the equilibrium; azimuthal component is IM2 in 3D and
  // IM3 in the 2D r-z geometry (matching ShearingBoxCC::SourceTermsCC)
  Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  // ideal gas: uniform pressure <problem>/pgas (the equilibrium needs no pressure
  // gradient; the radial forcing is the const_accel mimic), energy set below
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  const bool is_ideal = eos.is_ideal;
  Real pgas = is_ideal ? pin->GetOrAddReal("problem","pgas",1.0) : 0.0;
  Real gm1 = is_ideal ? (eos.gamma - 1.0) : 1.0;
  // optional smooth azimuthal modulation of the particle masses, m *= 1 + A sin(2 pi
  // (y - y_min)/L_y + phase): a deterministic (rank-independent) azimuthally STRUCTURED
  // dust distribution for exercising the shear-periodic deposit remap; 0 = inert
  Real mass_mod_amp = pin->GetOrAddReal("problem","mass_mod_amp",0.0);
  Real mass_mod_phase = pin->GetOrAddReal("problem","mass_mod_phase",0.0);
  Real mesh_y0 = pmy_mesh_->mesh_size.x2min;
  Real mesh_ly = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  bool three_d = pmy_mesh_->three_d;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->phydro->u0;
  // optional white-noise seed of the gas velocities (stability diagnostics: a uniform
  // state has no perturbation to grow, whatever the scheme's stability)
  Real gas_vpert = pin->GetOrAddReal("problem","gas_vpert",0.0);
  auto gids_ = pmbp->gids;
  par_for("nsh_gas", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real px = 0.0, py = 0.0, pz = 0.0;
    if (gas_vpert != 0.0) {
      std::uint64_t key = (static_cast<std::uint64_t>(gids_ + m) << 48)
                        ^ (static_cast<std::uint64_t>(k) << 32)
                        ^ (static_cast<std::uint64_t>(j) << 16)
                        ^ static_cast<std::uint64_t>(i);
      px = gas_vpert*(2.0*HashUniform01(key) - 1.0);
      py = gas_vpert*(2.0*HashUniform01(key + UINT64_C(0x9e3779b97f4a7c15)) - 1.0);
      pz = gas_vpert*(2.0*HashUniform01(key + UINT64_C(0x632be59bd9b4e019)) - 1.0);
    }
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = rho0*(ugx + px);
    u0(m,IM2,k,j,i) = three_d ? rho0*(ugp + py) : rho0*pz;
    u0(m,IM3,k,j,i) = three_d ? rho0*pz : rho0*(ugp + py);
    if (is_ideal) {
      u0(m,IEN,k,j,i) = pgas/gm1 + 0.5*rho0*(SQR(ugx) + SQR(ugp));
    }
  });

  // There are no particle or dust modules to initialize in the dust-free equilibrium.
  if (!dust_active) return;

  // Initialize particles at the per-species equilibrium velocities. The lattice is the
  // quiet start used by the NSH regression test. Random placement is the warm start for
  // nonlinear streaming-instability calculations (e.g. Johansen et al. 2007 Run BA).
  particles::Particles *ppar = pmbp->ppart;
  int npart = ppar->nprtcl_thispack;
  auto &mbsize = pmbp->pmb->mb_size;
  auto gids = pmbp->gids;
  int npart_permb = npart/nmb;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  int ppc_int = npart_permb/ncells;
  // finest-level lattice: per-block refinement factor, particle offsets, and a resized
  // particle array (the tags are re-created below)
  DualArray1D<int> lat_r("nsh_lat_r", nmb);
  DualArray1D<int> lat_off("nsh_lat_off", nmb+1);
  lat_off.h_view(0) = 0;
  for (int m=0; m<nmb; ++m) {
    int r = 1;
    if (lattice_finest) {
      r = 1 << (pmy_mesh_->max_level - pmbp->pmb->mb_lev.h_view(m));
    }
    lat_r.h_view(m) = r;
    int rd = r*r*(three_d ? r : 1);
    lat_off.h_view(m+1) = lat_off.h_view(m) + ppc_int*ncells*rd;
  }
  lat_r.template modify<HostMemSpace>();  lat_r.template sync<DevExeSpace>();
  lat_off.template modify<HostMemSpace>(); lat_off.template sync<DevExeSpace>();
  if (lattice_finest) {
    npart = lat_off.h_view(nmb);
    Kokkos::realloc(ppar->prtcl_rdata, ppar->nrdata, std::max(npart, 1));
    Kokkos::realloc(ppar->prtcl_idata, ppar->nidata, std::max(npart, 1));
    ppar->nprtcl_thispack = npart;
    Mesh *pm = pmy_mesh_;
    pm->nprtcl_thisrank = npart;
#if MPI_PARALLEL_ENABLED
    MPI_Allgather(&(pm->nprtcl_thisrank), 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT,
                  MPI_COMM_WORLD);
#else
    pm->nprtcl_eachrank[0] = npart;
#endif
    pm->nprtcl_total = 0;
    for (int n=0; n<global_variable::nranks; ++n) {
      pm->nprtcl_total += pm->nprtcl_eachrank[n];
    }
    ppar->CreateParticleTags(pin);
    if (global_variable::my_rank == 0) {
      std::cout << "# NSH lattice at the finest-level cell centres: " << pm->nprtcl_total
                << " particles" << std::endl;
    }
  }
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  if ((!random_placement &&
       ((npart_permb != ppc_int*ncells) || (ppc_int % nspec != 0))) ||
      (random_placement && (npart % nspec != 0))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH lattice placement requires <particles>/ppc to be an integer "
              << "multiple of <dust>/nspecies; random placement requires the total "
              << "number of particles per rank to be divisible by <dust>/nspecies"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int lnx1 = indcs.nx1, lnx2 = indcs.nx2, lnx3 = indcs.nx3;
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);
  std::uint64_t random_seed = static_cast<std::uint64_t>(
      pin->GetOrAddInteger("problem","random_seed",1));
  auto &taus_ = pmbp->pdust->taus;
  // per-species device table: [s][0]=v_x, [s][1]=v_phi, [s][2]=mass factor
  DualArray2D<Real> spdat("nsh_spdat",nspec,3);
  for (int s=0; s<nspec; ++s) {
    spdat.h_view(s,0) = vx[s];
    spdat.h_view(s,1) = vp[s];
    spdat.h_view(s,2) = eps[s]*rho0*static_cast<Real>(nspec)/ppc;
  }
  spdat.template modify<HostMemSpace>();
  spdat.template sync<DevExeSpace>();

  auto lat_r_ = lat_r.d_view;
  auto lat_off_ = lat_off.d_view;
  par_for("nsh_part", DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m, s;
    if (random_placement) {
      // Offset the tag stream before hashing so different seeds generate independent
      // position sets.  XOR with the raw tag is not sufficient: for consecutive tags
      // spanning complete power-of-two ranges it only permutes the same set of keys.
      std::uint64_t key = static_cast<std::uint64_t>(pi(PTAG,p))
                        + random_seed*UINT64_C(0x9e3779b97f4a7c15);
      m = static_cast<int>(HashUniform01(key)*static_cast<Real>(nmb));
      if (m > (nmb-1)) {m = nmb-1;}
      Real ux = HashUniform01(key + UINT64_C(0x632be59bd9b4e019));
      Real uy = HashUniform01(key + UINT64_C(0x8cb92baa3f3d8dd7));
      Real uz = HashUniform01(key + UINT64_C(0x58f38ded6f7c55b5));
      pr(IPX,p) = mbsize.d_view(m).x1min
                + ux*(mbsize.d_view(m).x1max - mbsize.d_view(m).x1min);
      pr(IPY,p) = mbsize.d_view(m).x2min
                + uy*(mbsize.d_view(m).x2max - mbsize.d_view(m).x2min);
      pr(IPZ,p) = three_d ? mbsize.d_view(m).x3min
                + uz*(mbsize.d_view(m).x3max - mbsize.d_view(m).x3min) : 0.0;
      // Stripe species by the local particle index. Unlike PTAG % nspec, this remains
      // balanced when <particles>/assign_tag=rank_order and nranks shares a factor with
      // nspec. The divisibility check above guarantees equal counts in this pack.
      s = p % nspec;
    } else {
      // lattice: block m owns particles [lat_off(m), lat_off(m+1)), ppc_int per cell of
      // its (r x) lattice grid
      m = 0;
      for (int mm=0; mm<nmb; ++mm) {if (p >= lat_off_(mm)) {m = mm;}}
      int r = lat_r_(m);
      int q = p - lat_off_(m);
      int c = q/ppc_int;
      s = (q % ppc_int) % nspec;
      int rnx1 = r*lnx1, rnx2 = r*lnx2, rnx3 = three_d ? r*lnx3 : lnx3;
      int i = c % rnx1;
      int j = (c/rnx1) % rnx2;
      int k = c/(rnx1*rnx2);
      pr(IPX,p) = CellCenterX(i, rnx1, mbsize.d_view(m).x1min,
                             mbsize.d_view(m).x1max);
      pr(IPY,p) = CellCenterX(j, rnx2, mbsize.d_view(m).x2min,
                             mbsize.d_view(m).x2max);
      pr(IPZ,p) = three_d ?
          CellCenterX(k, rnx3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max) : 0.0;
    }
    pi(PGID,p) = gids + m;
    pi(PSP,p) = s;
    pr(IPTS,p) = taus_.d_view(s);
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) {vol *= mbsize.d_view(m).dx3;}
    if (!random_placement && lat_r_(m) > 1) {
      int rd = lat_r_(m)*lat_r_(m)*(three_d ? lat_r_(m) : 1);
      vol /= static_cast<Real>(rd);
    }
    pr(IPM,p) = spdat.d_view(s,2)*vol;
    if (mass_mod_amp != 0.0) {
      pr(IPM,p) *= 1.0 + mass_mod_amp*sin(6.283185307179586*(pr(IPY,p) - mesh_y0)/mesh_ly
                                          + mass_mod_phase);
    }
    pr(IPVX,p) = spdat.d_view(s,0);
    pr(IPVY,p) = three_d ? spdat.d_view(s,1) : 0.0;
    pr(IPVZ,p) = three_d ? 0.0 : spdat.d_view(s,1);
    pr(IPRX,p) = 0.0;
    pr(IPRY,p) = 0.0;
    pr(IPRZ,p) = 0.0;
  });

  // particle timestep from the largest drift speed (transport is radial drift here)
  Real vmax = 1.0e-30;
  for (int s=0; s<nspec; ++s) {
    vmax = std::max(vmax, fabs(vx[s]));
    vmax = std::max(vmax, fabs(vp[s]));
  }
  Real dtnew = std::numeric_limits<float>::max();
  auto &msize = pmbp->pmb->mb_size;
  dtnew = std::min(dtnew, msize.h_view(0).dx1/vmax);
  dtnew = std::min(dtnew, msize.h_view(0).dx2/vmax);
  if (three_d) {dtnew = std::min(dtnew, msize.h_view(0).dx3/vmax);}
  ppar->dtnew = dtnew;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustNSHErrors()
//! \brief Measures the deviation of the mean gas and per-species dust drift velocities
//! from the NSH equilibrium at the end of the run and appends the errors to
//! "<basename>-errs.dat". Both the coupled and dust-free equilibria are fixed points,
//! so all errors should be at round-off level.

void DustNSHErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  int nspec = (pmbp->pdust != nullptr) ? pmbp->pdust->nspecies : 0;
  bool three_d = pm->three_d;

  Real omega0, qshear, etavk;
  std::vector<Real> eps, taus;
  ReadNSHParams(pin, pmbp, omega0, qshear, etavk, eps, taus);
  Real ugx, ugp;
  std::vector<Real> vx, vp;
  SolveNSH(omega0, qshear, 2.0*omega0*etavk, eps, taus, ugx, ugp, vx, vp);

  // mean gas velocities over active cells
  auto &indcs = pm->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &u0 = pmbp->phydro->u0;
  int iazim = three_d ? IM2 : IM3;
  Real gmx = 0.0, gmp = 0.0, gm = 0.0;
  Kokkos::parallel_reduce("nsh_gsum",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mx, Real &mp, Real &mass) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    mx += u0(m,IM1,k,j,i);
    mp += u0(m,iazim,k,j,i);
    mass += u0(m,IDN,k,j,i);
  }, Kokkos::Sum<Real>(gmx), Kokkos::Sum<Real>(gmp), Kokkos::Sum<Real>(gm));

  // per-species mean dust velocities (absent in the dust-free equilibrium)
  std::vector<Real> pvx(nspec), pvp(nspec), pn(nspec);
  if (nspec > 0) {
    auto &pr = ppar->prtcl_rdata;
    auto &pi = ppar->prtcl_idata;
    int npart = ppar->nprtcl_thispack;
    int ivazim = three_d ? IPVY : IPVZ;
    for (int s=0; s<nspec; ++s) {
      Real sx = 0.0, sp = 0.0, sn = 0.0;
      Kokkos::parallel_reduce("nsh_psum",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
      KOKKOS_LAMBDA(const int &p, Real &x_, Real &p_, Real &n_) {
        if (pi(PSP,p) == s) {
          x_ += pr(IPVX,p);
          p_ += pr(ivazim,p);
          n_ += 1.0;
        }
      }, Kokkos::Sum<Real>(sx), Kokkos::Sum<Real>(sp), Kokkos::Sum<Real>(sn));
      pvx[s] = sx;
      pvp[s] = sp;
      pn[s] = sn;
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &gmx, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gmp, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gm,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (nspec > 0) {
    MPI_Allreduce(MPI_IN_PLACE, pvx.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pvp.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pn.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif

  Real err_ugx = fabs(gmx/gm - ugx);
  Real err_ugp = fabs(gmp/gm - ugp);

  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      std::fclose(pfile);
      pfile = std::fopen(fname.c_str(), "a");
    } else {
      pfile = std::fopen(fname.c_str(), "w");
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  err_ugx  err_ugphi");
      if (nspec > 0) {std::fprintf(pfile, "  err_vx_s  err_vphi_s ...");}
      std::fprintf(pfile, "\n");
    }
    std::fprintf(pfile, "%04d  %04d  %04d  %05d  %e  %e", pm->mesh_indcs.nx1,
                 pm->mesh_indcs.nx2, pm->mesh_indcs.nx3, pm->ncycle, err_ugx, err_ugp);
    for (int s=0; s<nspec; ++s) {
      std::fprintf(pfile, "  %e  %e", fabs(pvx[s]/pn[s] - vx[s]),
                   fabs(pvp[s]/pn[s] - vp[s]));
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustNSHHistory()
//! \brief User history output: per-species sums of the radial and azimuthal dust drift
//! velocities and particle counts (summed over ranks; divide sums by counts for means).

void DustNSHHistory(HistoryData *pdata, Mesh *pm) {
  if (pm->pmb_pack->ppart == nullptr || pm->pmb_pack->pdust == nullptr) {
    pdata->nhist = 0;
    return;
  }
  particles::Particles *ppar = pm->pmb_pack->ppart;
  int nspec = pm->pmb_pack->pdust->nspecies;
  bool three_d = pm->three_d;
  int nsout = std::min(nspec, NHISTORY_VARIABLES/5);
  pdata->nhist = 5*nsout;

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  int ivazim = three_d ? IPVY : IPVZ;
  for (int s=0; s<nsout; ++s) {
    pdata->label[5*s  ] = "vxsum_" + std::to_string(s+1);
    pdata->label[5*s+1] = "vpsum_" + std::to_string(s+1);
    pdata->label[5*s+2] = "np_" + std::to_string(s+1);
    // mass-weighted sums: the species' momentum (Phase 4c momentum budgets)
    pdata->label[5*s+3] = "pxsum_" + std::to_string(s+1);
    pdata->label[5*s+4] = "ppsum_" + std::to_string(s+1);
    Real sx = 0.0, sp = 0.0, sn = 0.0, smx = 0.0, smp = 0.0;
    Kokkos::parallel_reduce("nsh_hist",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &x_, Real &p_, Real &n_, Real &mx_, Real &mp_) {
      if (pi(PSP,p) == s) {
        x_ += pr(IPVX,p);
        p_ += pr(ivazim,p);
        n_ += 1.0;
        mx_ += pr(IPM,p)*pr(IPVX,p);
        mp_ += pr(IPM,p)*pr(ivazim,p);
      }
    }, Kokkos::Sum<Real>(sx), Kokkos::Sum<Real>(sp), Kokkos::Sum<Real>(sn),
       Kokkos::Sum<Real>(smx), Kokkos::Sum<Real>(smp));
    pdata->hdata[5*s  ] = sx;
    pdata->hdata[5*s+1] = sp;
    pdata->hdata[5*s+2] = sn;
    pdata->hdata[5*s+3] = smx;
    pdata->hdata[5*s+4] = smp;
  }
  return;
}
