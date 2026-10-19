//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_growth.cpp
//! \brief Dust growth and fragmentation carried by the per-particle stopping time.
//!
//! Representative-particle convention: the mass IPM of a superparticle is fixed, and the
//! size of the grains it represents is carried by its stopping time IPTS (one size per
//! particle).  Under the constant-Stokes convention of shearing-box SI calculations, the
//! Stokes number S = Omega t_s is evaluated at the reference (midplane) gas density
//! rho_g0.  Epstein drag, t_s = rho_s a/(rho_g0 v_th) with v_th = sqrt(8/pi) c_s, and
//! equal-size sticking, da/dt = rho_d dv/rho_s (Birnstiel et al. 2012), give a law in
//! which the grain material density cancels:
//!     d t_s/dt = sqrt(pi/8) (rho_d/rho_g0) (dv/c_s)      [dS/dt = Omega d t_s/dt]
//! with rho_d the local PM dust density (all species) gathered at the particle, and
//!     dv^2 = dv_turb^2 + dv_drift^2,
//!     dv_turb  = sqrt(3 alpha_t S) c_s                (Ormel & Cuzzi 2007, intermediate)
//!     dv_drift = |v_p - u_g(x_p)|/2                   (vs a half-size partner)
//! where u_g is the gas velocity gathered with the drag kernel's stencil.  Where
//! dv > v_frag the grains fragment and shrink at the same rate (Vericel et al. 2021);
//! the turbulent term alone then caps S near S_frag = v_frag^2/(3 alpha_t c_s^2).
//!
//! The step is operator split: once per cycle, in "before_timeintegrator" before
//! GammaSwitch revalidates the stopping times and refreshes taus_max/taus_min, so IPTS is
//! never changed inside a step (the imex2+ drag history is consumed within one cycle).
//! The fractional change of t_s per cycle is capped at <dust>/growth_maxfrac, and S is
//! clamped to [growth_smin, growth_smax]; cumulative counts of particle-steps where
//! these limits bind are printed at the end of the run (DUST_GROWTH_SUMMARY).
//!
//! Requires <dust>/stopping_time_mode = dynamic.  Caveat: alpha_t must be consistent with
//! the box's actual turbulence; in an SI-only box (no imposed turbulence) a large alpha_t
//! makes the growth artificially fast.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::InitGrowth
//! \brief Parses and validates the <dust>/growth* parameters (called by the constructor
//! once the stopping-time mode, EOS and shearing-box parameters are known)

void DustGasDrag::InitGrowth(ParameterInput *pin) {
  std::string mode = pin->GetOrAddString("dust", "growth", "none");
  if (mode.compare("none") == 0) {
    growth = DustGrowth::none;
    return;
  } else if (mode.compare("monodisperse") == 0) {
    growth = DustGrowth::monodisperse;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/growth = '" << mode << "' not recognized (must be none or "
              << "monodisperse)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (stopping_time_mode != DustStoppingTimeMode::dynamic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/growth = " << mode << " changes the particle stopping times and "
              << "requires <dust>/stopping_time_mode = dynamic" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  growth_alpha   = pin->GetReal("dust", "growth_alpha");
  growth_vfrag   = pin->GetReal("dust", "growth_vfrag");     // in units of c_s
  growth_smin    = pin->GetOrAddReal("dust", "growth_smin", 1.0e-5);
  growth_smax    = pin->GetOrAddReal("dust", "growth_smax", 1.0);
  growth_maxfrac = pin->GetOrAddReal("dust", "growth_maxfrac", 0.05);
  growth_drift   = pin->GetOrAddBoolean("dust", "growth_drift", true);
  // reference gas density of the Stokes number and the units of dv
  growth_rho0    = pin->GetOrAddReal("dust", "growth_rho0", 1.0);
  growth_omega   = pin->GetOrAddReal("dust", "growth_omega",
                                     is_shearing_box ? omega0 : 1.0);
  if (gas_ideal) {
    growth_cs = pin->GetReal("dust", "growth_cs");
  } else {
    growth_cs = pin->GetOrAddReal("dust", "growth_cs",
                                  pmy_pack->phydro->peos->eos_data.iso_cs);
  }
  if (!(growth_alpha >= 0.0) || !(growth_vfrag > 0.0) || !(growth_smin > 0.0) ||
      !(growth_smax > growth_smin) || !(growth_maxfrac > 0.0) ||
      !(growth_rho0 > 0.0) || !(growth_omega > 0.0) || !(growth_cs > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Invalid <dust>/growth parameters: need growth_alpha >= 0, "
              << "growth_vfrag > 0, 0 < growth_smin < growth_smax, growth_maxfrac > 0, "
              << "and positive growth_rho0, growth_omega, growth_cs" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(6)
              << "# dust: growth = monodisperse: d t_s/dt = sqrt(pi/8) (rho_d/rho_g0) "
              << "dv/c_s, dv^2 = 3 alpha_t S c_s^2"
              << (growth_drift ? " + |v_p - u_g|^2/4" : "")
              << "; alpha_t = " << growth_alpha << ", v_frag = " << growth_vfrag
              << " c_s";
    if (growth_alpha > 0.0) {
      std::cout << " (turbulent cap S_frag = "
                << SQR(growth_vfrag)/(3.0*growth_alpha) << ")";
    }
    std::cout << ", S in [" << growth_smin << ", " << growth_smax
              << "], max fractional change " << growth_maxfrac << " per cycle"
              << " (rho_g0 = " << growth_rho0 << ", Omega = " << growth_omega
              << ", c_s = " << growth_cs << ")" << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::GrowthStep
//! \brief "before_timeintegrator" task: assemble the PM dust density with its ghost
//! layer, then advance every particle's stopping time over the coming step dt

TaskStatus DustGasDrag::GrowthStep(Driver *pdrive, int stage) {
  if (growth == DustGrowth::none) {return TaskStatus::complete;}
  // collective (ghost exchanges): every rank, every cycle, even without particles
  AssembleDustDensityNow();
  GrowStoppingTimes(pmy_pack->pmesh->dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::GrowStoppingTimes
//! \brief The growth kernel: gathers rho_d (rho_dust, ghost-filled) and the gas velocity
//! (w0, valid in the ghosts at the start of a cycle) with the drag kernels' PM stencil

void DustGasDrag::GrowStoppingTimes(Real dt) {
  ++growth_steps;
  particles::Particles *ppar = pmy_pack->ppart;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) {return;}

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &w0 = pmy_pack->phydro->w0;
  auto &rhod = rho_dust;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto gids = pmy_pack->gids;
  int scheme = static_cast<int>(deposit);

  const Real coef = std::sqrt(M_PI/8.0)/(growth_rho0*growth_cs);  // d t_s/dt per rho_d dv
  const Real turb = 3.0*growth_alpha*growth_omega*SQR(growth_cs);  // dv_turb^2 per t_s
  const Real vfrag = growth_vfrag*growth_cs;
  const Real tsmin = growth_smin/growth_omega, tsmax = growth_smax/growth_omega;
  const Real maxfrac = growth_maxfrac;
  const bool drift = growth_drift;
  // 3D shearing box without orbital advection: w0 carries the background shear, the
  // particle velocities do not (the shift of ShiftGasFrame, at each stencil cell)
  const bool full = gas_full_frame;
  const Real qo = qshear*omega0;

  int ncap = 0, nclamp = 0, nfrag = 0;
  Kokkos::parallel_reduce("dust_growth", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &cap, int &clamp, int &frag) {
    int m = pi(PGID,p) - gids;
    int ip, jp, kp;
    Real wx[3], wy[3], wz[3];
    PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
              scheme, ip, wx);
    PMWeights(pr(IPY,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
              scheme, jp, wy);
    if (three_d) {
      PMWeights(pr(IPZ,p), mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, nx3, ks,
                scheme, kp, wz);
    } else {
      kp = ks;
      wz[0] = 0.0; wz[1] = 1.0; wz[2] = 0.0;
    }
    Real dx1 = (mbsize.d_view(m).x1max - mbsize.d_view(m).x1min)/static_cast<Real>(nx1);
    Real rd = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
    int clo = three_d ? 0 : 1, chi = three_d ? 2 : 1;
    for (int c=clo; c<=chi; ++c) {
      for (int b=0; b<3; ++b) {
        Real wcb = wz[c]*wy[b];
        if (wcb == 0.0) continue;
        for (int a=0; a<3; ++a) {
          Real w = wcb*wx[a];
          int kk = kp+c-1, jj = jp+b-1, ii = ip+a-1;
          rd += w*rhod(m,0,kk,jj,ii);
          ux += w*w0(m,IVX,kk,jj,ii);
          Real vy = w0(m,IVY,kk,jj,ii);
          if (full) {
            vy += qo*(mbsize.d_view(m).x1min + (static_cast<Real>(ii - is) + 0.5)*dx1);
          }
          uy += w*vy;
          uz += w*w0(m,IVZ,kk,jj,ii);
        }
      }
    }
    Real ts = pr(IPTS,p);
    Real dv2 = turb*ts;
    if (drift) {
      dv2 += 0.25*(SQR(pr(IPVX,p) - ux) + SQR(pr(IPVY,p) - uy) + SQR(pr(IPVZ,p) - uz));
    }
    Real dv = sqrt(dv2);
    Real rate = coef*fmax(rd, 0.0)*dv;
    if (dv > vfrag) {
      rate = -rate;
      frag += 1;
    }
    Real dts = rate*dt;
    Real lim = maxfrac*ts;
    if (fabs(dts) > lim) {
      dts = (dts > 0.0) ? lim : -lim;
      cap += 1;
    }
    Real tsn = ts + dts;
    if (tsn < tsmin || tsn > tsmax) {
      tsn = fmin(fmax(tsn, tsmin), tsmax);
      clamp += 1;
    }
    pr(IPTS,p) = tsn;
  }, Kokkos::Sum<int>(ncap), Kokkos::Sum<int>(nclamp), Kokkos::Sum<int>(nfrag));
  growth_capped += static_cast<unsigned long long>(ncap);
  growth_clamped += static_cast<unsigned long long>(nclamp);
  growth_frag += static_cast<unsigned long long>(nfrag);
}

} // namespace dust
