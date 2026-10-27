//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file jeans_wave.cpp
//  \brief Problem generator for Jeans wave test with self-gravity
//
//  Tests the linear dispersion relation for sound+gravity waves in a uniform medium.
//  For wave vector k and sound speed cs, the dispersion relation is:
//    omega^2 = k^2*cs^2 - 4*pi*G*rho0
//  The Jeans wavenumber is k_J = sqrt(4*pi*G*rho0)/cs
//  For k < k_J: exponential growth (gravitational instability)
//  For k > k_J: oscillatory waves (modified sound waves)
//
//  References:
//  - Jeans (1902) Phil. Trans. R. Soc. Lond. A 199, 1-53
//  - Stone et al. (2008) ApJS 178, 137-177 (Athena test suite)
//
//  Magneto-Jeans mode (MHD with <problem>/b_angle given; 2026-10-26).  A uniform field
//  B0 = b0 (cos(theta) k_hat + sin(theta) e_perp), theta = b_angle (degrees) between B0
//  and k, e_perp = (-sin a3, cos a3, 0) the unit vector perpendicular to k in the plane of
//  B0.  Linearizing ideal (or isothermal) MHD with self-gravity about it, a compressive
//  wave with the perturbed field along e_perp obeys (Chandrasekhar 1961, ch. XIII)
//      (w^2 - W_J^2)(w^2 - k^2 vA^2 cos^2 theta) = k^2 vA^2 sin^2 theta w^2,
//      W_J^2 = k^2 cs^2 - 4 pi G rho0,  vA^2 = b0^2/rho0,
//  whose roots are the slow (Jeans) branch w_-^2 and the fast branch w_+^2:
//      theta = 0 :  w^2 = W_J^2 (the field plays no part)
//      theta = 90:  w^2 = k^2 (cs^2 + vA^2) - 4 pi G rho0 (the field stiffens the gas)
//  and the slow branch is unstable whenever W_J^2 < 0, at any theta < 90 degrees.  The
//  initial condition is the eigenmode of the branch <problem>/mode = slow | fast (the
//  smaller | larger root; at theta = 0 and 90 the single compressive root above):
//      drho = rho0 A sin(k xi),  dB_perp = C drho,  C = B_perp w^2/(rho0 (w^2 - k^2 vA_par^2)),
//      stable (w^2 > 0): no velocity -> a standing wave, drho ~ cos(w t);
//      unstable (w^2 = -s^2): dv_par = (s/k) A cos(k xi), dv_perp = B_par C A (k/s) cos(k xi)
//                             -> pure growth, drho ~ exp(s t);
//  with xi = k_hat.x and the pressure perturbation isentropic (dp = cs^2 drho).  The
//  perturbed field comes from a vector potential on the cell edges, so div B = 0 to round
//  off.  <problem>/wave_dir = x1 puts k along x1 (lambda = Lx1; a thin 3-D box is a cheap
//  1-D test); the default (oblique) keeps the original box-diagonal wave.  The user history
//  (<problem>/user_hist = true) projects the solution onto the mode: a_sin, a_cos (density),
//  bp_sin (dB_perp), vpar_cos, vperp_cos (velocities), each (2/V) int (.) sin|cos(k xi) dV,
//  then divb = int |div B| dx_min dV (div B from the face fields), babs = int |B| dV and
//  nmb = the number of MeshBlocks.
//
//  Refinement (<mesh_refinement>/refinement = adaptive): <problem>/amr_mode = jeans (the
//  default: refine where the Jeans length has fewer than njeans_amr cells) or window: refine
//  every MeshBlock whose x1 extent overlaps the interval |x1 - xc(t)| < win_hwidth, with
//  xc(t) = win_x0 + win_speed t wrapped into the box, and derefine all others -- a slab
//  swept through the wave so that blocks are created and destroyed while it runs (2026-10-26).

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
  Real jw_iso_cs = 1.0;
  Real jw_four_pi_G = 1.0;
  Real jw_njeans_threshold = 16.0;
  Real jw_njeans_derefine = 2.5;
  Real jw_rho0 = 1.0;
  Real jw_amp = 1.0e-6;
  Real jw_k_wave = 0.0;
  Real jw_omega = 0.0;
  Real jw_n_jeans = -1.0;
  Real jw_cos_a2 = 1.0;
  Real jw_cos_a3 = 1.0;
  Real jw_sin_a2 = 0.0;
  Real jw_sin_a3 = 0.0;
  Real jw_v0 = 0.0;
  Real jw_p0 = 1.0;
  Real jw_gamma = 5.0/3.0;
  bool jw_perturb_iso = false;   // isothermal (Tomida & Stone 4.3) vs isentropic
  bool jw_ts_errors = false;     // print the TS43 L1-error line at the end
  bool jw_is_ideal = false;
  bool jw_unstable = false;      // the excited mode grows (w^2 < 0)
  Real jw_khat[3] = {1.0, 0.0, 0.0};
  Real jw_eperp[3] = {0.0, 1.0, 0.0};
  Real jw_bperp0 = 0.0;          // background field component along e_perp
  // moving-window refinement (<problem>/amr_mode = window)
  Real jw_win_x0 = 0.5, jw_win_hw = 0.125, jw_win_speed = 0.0;
  Real jw_x1min = 0.0, jw_lx1 = 1.0;
}  // namespace

void JeansWaveHistory(HistoryData *pdata, Mesh *pm);

// vector potential of the magneto-Jeans perturbation, A = a(xi) n_hat with
// a = -(dB/k) cos(k xi) and n_hat = e_perp x k_hat, so that curl A = dB e_perp sin(k xi)
KOKKOS_INLINE_FUNCTION
Real JWVecPot(const int comp, const Real x1, const Real x2, const Real x3,
              const Real k1, const Real k2, const Real k3, const Real n1, const Real n2,
              const Real n3, const Real kw, const Real db) {
  Real a = -(db/kw)*cos(kw*(k1*x1 + k2*x2 + k3*x3));
  return (comp == 1) ? a*n1 : ((comp == 2) ? a*n2 : a*n3);
}

void JeansWaveRefinement(MeshBlockPack *pmbp);
void JeansWaveWindowRefinement(MeshBlockPack *pmbp);
void JeansWaveErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SelfGravity()
//  \brief Problem Generator for Jeans instability/wave test
//
//  Sets up a sinusoidal density perturbation in a uniform medium:
//    rho = rho0 * (1 + amp * sin(k*x))
//    v = v0 * sin(k*x)  [optionally for traveling wave]
//    P = P0 * (1 + amp * sin(k*x)) [for isentropic perturbation]
//
//  For periodic domain with box size L, wavelength lambda = L/n_waves:
//    k = 2*pi*n_waves/L
//  Compare k to k_J = sqrt(4*pi*G*rho0)/cs to predict behavior

void ProblemGenerator::SelfGravity(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool use_mhd = (pmbp->pmhd != nullptr);
  std::string soe = use_mhd ? "mhd" : "hydro";

  // --- AMR Jeans criterion (must be set on both fresh start and restart) ---
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real n_jeans = pin->GetOrAddReal("problem", "n_jeans", -1.0);
  std::string eos_type = pin->GetString(soe, "eos");
  Real cs;
  Real gamma = 0.0, gm1 = 0.0, p0 = 0.0;
  bool is_isothermal = (eos_type == "isothermal");
  if (is_isothermal) {
    cs = pin->GetReal(soe, "iso_sound_speed");
  } else {
    gamma = pin->GetOrAddReal(soe, "gamma", 5.0/3.0);
    gm1 = gamma - 1.0;
    p0 = pin->GetOrAddReal("problem", "p0", 1.0);
    cs = std::sqrt(gamma * p0 / rho0);
  }

  jw_four_pi_G = four_pi_G;
  jw_iso_cs = cs;
  jw_njeans_threshold = pin->GetOrAddReal("problem", "njeans_amr", 16.0);
  jw_njeans_derefine = pin->GetOrAddReal("problem", "njeans_derefine", 2.5);
  std::string amr_mode = pin->GetOrAddString("problem", "amr_mode", "jeans");
  if (amr_mode == "window") {
    jw_win_x0 = pin->GetOrAddReal("problem", "win_x0", 0.5);
    jw_win_hw = pin->GetOrAddReal("problem", "win_hwidth", 0.125);
    jw_win_speed = pin->GetOrAddReal("problem", "win_speed", 0.0);
    jw_x1min = pin->GetReal("mesh", "x1min");
    jw_lx1 = pin->GetReal("mesh", "x1max") - jw_x1min;
    user_ref_func = JeansWaveWindowRefinement;
  } else if (amr_mode == "jeans") {
    user_ref_func = JeansWaveRefinement;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem>/amr_mode = " << amr_mode << " (use jeans or window)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pgen_final_func = JeansWaveErrors;

  if (restart) return;

  Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-6);
  Real v0 = pin->GetOrAddReal("problem", "v0", 0.0);

  // pressure-perturbation type: "isentropic" (the acoustic eigenmode, default) or
  // "isothermal" (delta_p = p0*A*sin kx, the Tomida & Stone 2023 sec. 4.3 setup,
  // which excites a mix of the gravito-acoustic wave and a static self-gravitating
  // mode). Irrelevant for an isothermal EOS.
  std::string perturb = pin->GetOrAddString("problem", "perturb", "isentropic");
  bool perturb_iso = (perturb == "isothermal");
  jw_perturb_iso = perturb_iso;
  jw_ts_errors = pin->GetOrAddBoolean("problem", "ts_errors", false);
  jw_p0 = p0;
  jw_gamma = gamma;
  jw_is_ideal = !is_isothermal;

  // Get domain size to compute actual wavenumber
  Real Lx1 = pin->GetReal("mesh", "x1max") - pin->GetReal("mesh", "x1min");
  Real Lx2 = pin->GetReal("mesh", "x2max") - pin->GetReal("mesh", "x2min");
  Real Lx3 = pin->GetReal("mesh", "x3max") - pin->GetReal("mesh", "x3min");

  // wave direction: the box diagonal (original test) or x1
  std::string wave_dir = pin->GetOrAddString("problem", "wave_dir", "oblique");
  Real ang_3 = (wave_dir == "x1") ? 0.0 : std::atan(Lx1/Lx2);
  Real sin_a3 = std::sin(ang_3);
  Real cos_a3 = std::cos(ang_3);
  Real ang_2 = (wave_dir == "x1") ? 0.0 : std::atan(0.5*(Lx1*cos_a3 + Lx2*sin_a3)/Lx3);
  Real sin_a2 = std::sin(ang_2);
  Real cos_a2 = std::cos(ang_2);

  Real x1 = Lx1*cos_a2*cos_a3;
  Real x2 = Lx2*cos_a2*sin_a3;
  Real x3 = Lx3*sin_a2;

  Real lambda = (wave_dir == "x1") ? Lx1 : std::min(x1, std::min(x2, x3));
  Real lambda_jeans = lambda/n_jeans;
  // If n_jeans is specified, set effective four_pi_G to enforce k_J = n_jeans * k
  if (n_jeans > 0.0) {
    Real G = M_PI * (cs*cs)/(rho0*lambda_jeans*lambda_jeans);
    four_pi_G = 4 * M_PI * G;
    pin->SetReal("gravity", "four_pi_G", four_pi_G);
    if (pmy_mesh_->pmb_pack->pgrav != nullptr) {
      pmy_mesh_->pmb_pack->pgrav->four_pi_G = four_pi_G;
      if (pmy_mesh_->pmb_pack->pgrav->pmgd != nullptr) {
        pmy_mesh_->pmb_pack->pgrav->pmgd->SetFourPiG(four_pi_G);
      }
    }
    jw_four_pi_G = four_pi_G;
  }
  Real k_wave = 2.0*M_PI/lambda;
  Real k_jeans = 2.0*M_PI/lambda_jeans;
  Real omega2 = SQR(k_wave)*cs*cs*(1.0 - SQR(n_jeans));
  if (n_jeans <= 0.0) omega2 = SQR(k_wave)*cs*cs - four_pi_G*rho0;

  // magneto-Jeans mode (see the file header)
  bool mj_mode = use_mhd && pin->DoesParameterExist("problem", "b_angle");
  Real b0_mag = 0.0, theta = 0.0, b_par = 0.0, b_perp = 0.0, c_db = 0.0;
  Real khat[3] = {cos_a2*cos_a3, cos_a2*sin_a3, sin_a2};
  Real eperp[3] = {-sin_a3, cos_a3, 0.0};
  if (mj_mode) {
    b0_mag = pin->GetOrAddReal("problem", "b0", 0.0);
    theta = pin->GetReal("problem", "b_angle")*M_PI/180.0;
    std::string mode = pin->GetOrAddString("problem", "mode", "slow");
    b_par = b0_mag*std::cos(theta);
    b_perp = b0_mag*std::sin(theta);
    Real k2 = SQR(k_wave);
    Real wj2 = k2*cs*cs - four_pi_G*rho0;
    Real va2 = SQR(b0_mag)/rho0, vapar2 = SQR(b_par)/rho0;
    Real sum = wj2 + k2*va2, prod = wj2*k2*vapar2;
    Real disc = std::sqrt(std::fmax(sum*sum - 4.0*prod, 0.0));
    // the two roots carry density except at theta = 0 and 90 degrees, where one of them
    // is not compressive (k^2 vA^2: a transverse wave; 0: a static mode) and the
    // compressive one is taken whatever <problem>/mode says
    if (std::fabs(b_perp) < 1.0e-12*std::fmax(b0_mag, 1.0)) {
      omega2 = wj2;              // B0 along k: the plain Jeans mode
    } else if (std::fabs(b_par) < 1.0e-12*std::fmax(b0_mag, 1.0)) {
      omega2 = sum;              // B0 across k: k^2 (cs^2 + vA^2) - 4 pi G rho0
    } else {
      omega2 = (mode == "fast") ? 0.5*(sum + disc) : 0.5*(sum - disc);
    }
    Real den = rho0*(omega2 - k2*vapar2);
    c_db = (std::fabs(den) > 0.0) ? b_perp*omega2/den : 0.0;
    if (global_variable::my_rank == 0) {
      std::cout << "Magneto-Jeans mode: b0 = " << b0_mag << ", theta = "
                << theta*180.0/M_PI << " deg, vA/cs = " << std::sqrt(va2)/cs
                << ", mode = " << mode << ", w^2 = " << omega2
                << " (W_J^2 = " << wj2 << "), dB_perp/drho = " << c_db << std::endl;
    }
    if (!is_isothermal && perturb_iso && global_variable::my_rank == 0) {
      std::cout << "### WARNING: perturb = isothermal is not the magneto-Jeans "
                << "eigenmode (it assumes dp = cs^2 drho)" << std::endl;
    }
  }
  bool unstable = mj_mode ? (omega2 < 0.0) : (n_jeans > 1.0);
  Real omega = std::sqrt(std::abs(omega2));
  for (int d = 0; d < 3; ++d) {jw_khat[d] = khat[d]; jw_eperp[d] = eperp[d];}
  jw_unstable = unstable;
  jw_bperp0 = b_perp;
  user_hist_func = JeansWaveHistory;

  if (global_variable::my_rank == 0) {
    std::cout << "Jeans wave test parameters:" << std::endl;
    std::cout << "  rho0 = " << rho0 << ", cs = " << cs;
    if (!is_isothermal) std::cout << ", p0 = " << p0;
    std::cout << std::endl;
    std::cout << "  k_Jeans = " << k_jeans << ", k_wave = " << k_wave << std::endl;
    std::cout << "  k/k_J = " << k_wave/k_jeans << std::endl;
    if (n_jeans > 0.0) {
      std::cout << "  n_Jeans = " << n_jeans << " (lambda/lambda_Jeans)" << std::endl;
    }
    std::cout << "  four_pi_G = " << four_pi_G << std::endl;
    if (v0 != 0.0) {
      std::cout << "  Background velocity v0 = " << v0 << std::endl;
    }
    if (unstable) {
      std::cout << "  Gravitationally unstable! Growth rate = " << omega << std::endl;
    } else {
      std::cout << "  Stable oscillation. Frequency = " << omega << std::endl;
    }
  }

  // Store parameters for JeansWaveErrors
  jw_rho0 = rho0;
  jw_amp = amp;
  jw_k_wave = k_wave;
  jw_omega = omega;
  jw_n_jeans = n_jeans;
  jw_cos_a2 = cos_a2;
  jw_cos_a3 = cos_a3;
  jw_sin_a2 = sin_a2;
  jw_sin_a3 = sin_a3;
  jw_v0 = v0;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  Real pfac = perturb_iso ? 1.0 : gamma;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;

    par_for("jeans_wave_hydro_init", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real x = cos_a2*(x1v*cos_a3 + x2v*sin_a3) + x3v*sin_a2;
      Real sinkx = std::sin(x*k_wave);
      Real coskx = std::cos(x*k_wave);
      Real M = (n_jeans > 1.0) ? rho0*(omega/k_wave)*amp*coskx : 0.0;

      Real dens = rho0*(1.0+amp*sinkx+amp*amp*std::sin(x1v*k_wave));
      u0(m, IDN, k, j, i) = dens;
      u0(m, IM1, k, j, i) = M*cos_a3*cos_a2 + dens*v0*cos_a3*cos_a2;
      u0(m, IM2, k, j, i) = M*sin_a3*cos_a2 + dens*v0*sin_a3*cos_a2;
      u0(m, IM3, k, j, i) = M*sin_a2         + dens*v0*sin_a2;
      if (!is_isothermal) {
        u0(m, IEN, k, j, i) = p0/gm1*(1.0 + pfac*amp*sinkx);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM1, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM2, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM3, k, j, i))/u0(m, IDN, k, j, i);
      }
    });

  // Initialize MHD variables ---------------------------------
  } else if (pmbp->pmhd != nullptr && mj_mode) {
    // magneto-Jeans eigenmode (file header)
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    Real kx1 = khat[0], kx2 = khat[1], kx3 = khat[2];
    Real ep1 = eperp[0], ep2 = eperp[1], ep3 = eperp[2];
    // n_hat = e_perp x k_hat
    Real nh1 = ep2*kx3 - ep3*kx2, nh2 = ep3*kx1 - ep1*kx3, nh3 = ep1*kx2 - ep2*kx1;
    Real bb1 = b_par*kx1 + b_perp*ep1, bb2 = b_par*kx2 + b_perp*ep2;
    Real bb3 = b_par*kx3 + b_perp*ep3;
    Real db = c_db*rho0*amp;                  // dB_perp amplitude
    Real s_gr = unstable ? omega : 0.0;
    Real vpar_a = unstable ? (s_gr/k_wave)*amp : 0.0;
    Real vperp_a = unstable ? b_par*c_db*amp*k_wave/s_gr : 0.0;
    Real kw = k_wave;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;

    par_for("mjeans_b", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
      Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;
      Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max), x1fp = x1f + dx1;
      Real x2f = LeftEdgeX(j-js, nx2, x2min, x2max), x2fp = x2f + dx2;
      Real x3f = LeftEdgeX(k-ks, nx3, x3min, x3max), x3fp = x3f + dx3;
      auto A = [=](int c, Real y1, Real y2, Real y3) {
        return JWVecPot(c, y1, y2, y3, kx1, kx2, kx3, nh1, nh2, nh3, kw, db);
      };
      // faces at the lower edges of this cell (and the upper ones on the last cell)
      b0.x1f(m,k,j,i) = bb1 + (A(3, x1f, x2fp, x3v) - A(3, x1f, x2f, x3v))/dx2
                            - (A(2, x1f, x2v, x3fp) - A(2, x1f, x2v, x3f))/dx3;
      b0.x2f(m,k,j,i) = bb2 + (A(1, x1v, x2f, x3fp) - A(1, x1v, x2f, x3f))/dx3
                            - (A(3, x1fp, x2f, x3v) - A(3, x1f, x2f, x3v))/dx1;
      b0.x3f(m,k,j,i) = bb3 + (A(2, x1fp, x2v, x3f) - A(2, x1f, x2v, x3f))/dx1
                            - (A(1, x1v, x2fp, x3f) - A(1, x1v, x2f, x3f))/dx2;
      if (i == ie) {
        b0.x1f(m,k,j,i+1) = bb1 + (A(3, x1fp, x2fp, x3v) - A(3, x1fp, x2f, x3v))/dx2
                                - (A(2, x1fp, x2v, x3fp) - A(2, x1fp, x2v, x3f))/dx3;
      }
      if (j == je) {
        b0.x2f(m,k,j+1,i) = bb2 + (A(1, x1v, x2fp, x3fp) - A(1, x1v, x2fp, x3f))/dx3
                                - (A(3, x1fp, x2fp, x3v) - A(3, x1f, x2fp, x3v))/dx1;
      }
      if (k == ke) {
        b0.x3f(m,k+1,j,i) = bb3 + (A(2, x1fp, x2v, x3fp) - A(2, x1f, x2v, x3fp))/dx1
                                - (A(1, x1v, x2fp, x3fp) - A(1, x1v, x2f, x3fp))/dx2;
      }
    });

    par_for("mjeans_u", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real xi = kx1*x1v + kx2*x2v + kx3*x3v;
      Real sinkx = sin(kw*xi), coskx = cos(kw*xi);
      Real dens = rho0*(1.0 + amp*sinkx);
      Real vpar = vpar_a*coskx + v0, vperp = vperp_a*coskx;
      Real v1 = vpar*kx1 + vperp*ep1, v2 = vpar*kx2 + vperp*ep2, v3 = vpar*kx3 + vperp*ep3;
      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = dens*v1;
      u0(m,IM2,k,j,i) = dens*v2;
      u0(m,IM3,k,j,i) = dens*v3;
      Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      Real bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
      bcc0(m,IBX,k,j,i) = bx; bcc0(m,IBY,k,j,i) = by; bcc0(m,IBZ,k,j,i) = bz;
      if (!is_isothermal) {
        u0(m,IEN,k,j,i) = p0/gm1*(1.0 + gamma*amp*sinkx)
                          + 0.5*dens*(v1*v1 + v2*v2 + v3*v3)
                          + 0.5*(bx*bx + by*by + bz*bz);
      }
    });

  // Initialize MHD variables (field along x1; the original test) ---------------
  } else if (pmbp->pmhd != nullptr) {
    auto &u0 = pmbp->pmhd->u0;
    Real b0_val = pin->GetOrAddReal("problem", "b0", 0.0);

    par_for("jeans_wave_mhd_init", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real x = cos_a2*(x1v*cos_a3 + x2v*sin_a3) + x3v*sin_a2;
      Real sinkx = std::sin(x*k_wave);
      Real coskx = std::cos(x*k_wave);
      Real M = (n_jeans > 1.0) ? rho0*(omega/k_wave)*amp*coskx : 0.0;

      Real dens = rho0*(1.0+amp*sinkx+amp*amp*std::sin(x1v*k_wave));
      u0(m, IDN, k, j, i) = dens;
      u0(m, IM1, k, j, i) = M*cos_a3*cos_a2 + dens*v0*cos_a3*cos_a2;
      u0(m, IM2, k, j, i) = M*sin_a3*cos_a2 + dens*v0*sin_a3*cos_a2;
      u0(m, IM3, k, j, i) = M*sin_a2         + dens*v0*sin_a2;
      if (!is_isothermal) {
        u0(m, IEN, k, j, i) = p0/gm1*(1.0 + pfac*amp*sinkx);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM1, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM2, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM3, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(b0_val);
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("jeans_wave_bfield", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = b0_val;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;
      if (i==ie) b0.x1f(m,k,j,i+1) = b0_val;
      if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
      if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
      bcc0(m,IBX,k,j,i) = b0_val;
      bcc0(m,IBY,k,j,i) = 0.0;
      bcc0(m,IBZ,k,j,i) = 0.0;
    });
  }

  // optional white-noise velocity seed, dv_i = noise_amp*(r - 1/2) per component with r a
  // hash of the cell position (the same on every mesh and decomposition): breaks the
  // symmetry of the quasi-1-D runs so that numerical modes across x2/x3 can be measured
  // on equal terms with any solver
  Real noise = pin->GetOrAddReal("problem", "noise_amp", 0.0);
  if (noise > 0.0) {
    auto &u0n = use_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
    bool ideal = !is_isothermal;
    par_for("jeans_wave_noise", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real d = u0n(m,IDN,k,j,i);
      Real ke0 = 0.5*(SQR(u0n(m,IM1,k,j,i)) + SQR(u0n(m,IM2,k,j,i)) +
                      SQR(u0n(m,IM3,k,j,i)))/d;
      for (int n=0; n<3; ++n) {
        Real h = sin(x1v*(12.9898 + 7.0*n) + x2v*(78.233 + 3.0*n) + x3v*(37.719 + 5.0*n))
                 *43758.5453;
        Real r = h - floor(h);
        u0n(m,IM1+n,k,j,i) += d*noise*(r - 0.5);
      }
      if (ideal) {
        u0n(m,IEN,k,j,i) += 0.5*(SQR(u0n(m,IM1,k,j,i)) + SQR(u0n(m,IM2,k,j,i)) +
                                 SQR(u0n(m,IM3,k,j,i)))/d - ke0;
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void JeansWaveRefinement()
//! \brief Jeans-length AMR criterion for the Jeans wave test.
//!
//! Computes the minimum number of cells per Jeans length in each MeshBlock:
//!   nJ = 2*pi*cs / (dx * sqrt(four_pi_G * rho_max))
//! Refines if nJ < threshold, derefines if nJ > 2.5*threshold.

void JeansWaveRefinement(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  DvceArray5D<Real> u0;
  if (pmbp->phydro != nullptr) {
    u0 = pmbp->phydro->u0;
  } else {
    u0 = pmbp->pmhd->u0;
  }
  auto &size = pmbp->pmb->mb_size;
  Real cs = jw_iso_cs;
  Real fpG = jw_four_pi_G;
  Real njeans = jw_njeans_threshold;
  Real derefine = jw_njeans_derefine;

  par_for_outer("JeansWaveAMR", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_rhomax;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [&](const int idx, Real &rhomax) {
        int k = (idx) / nji;
        int j = (idx - k * nji) / nx1;
        int i = (idx - k * nji - j * nx1) + is;
        j += js;
        k += ks;
        rhomax = Kokkos::fmax(u0(m, IDN, k, j, i), rhomax);
      },
      Kokkos::Max<Real>(team_rhomax));

    Real dx = size.d_view(m).dx1;
    Real nj_min = 2.0 * M_PI * cs / (dx * Kokkos::sqrt(fpG * team_rhomax));

    if (nj_min < njeans) {
      refine_flag.d_view(m + mbs) = 1;
    } else if (nj_min > njeans * derefine) {
      refine_flag.d_view(m + mbs) = -1;
    } else {
      refine_flag.d_view(m + mbs) = 0;
    }
  });

  refine_flag.modify_device();
  refine_flag.sync_host();
}

//----------------------------------------------------------------------------------------
//! \fn void JeansWaveWindowRefinement()
//! \brief Moving-window AMR criterion: refine every MeshBlock whose x1 extent overlaps
//! |x1 - xc(t)| < win_hwidth (periodic in x1), xc(t) = win_x0 + win_speed t, and derefine
//! all others. The window spans x2 and x3. Host data only (MeshBlock bounds).

void JeansWaveWindowRefinement(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  auto &size = pmbp->pmb->mb_size;
  Real xc = jw_win_x0 + jw_win_speed*(pmbp->pmesh->time) - jw_x1min;
  xc = jw_x1min + (xc - jw_lx1*std::floor(xc/jw_lx1));
  for (int m=0; m<(pmbp->nmb_thispack); ++m) {
    Real x1min = size.h_view(m).x1min, x1max = size.h_view(m).x1max;
    bool in = false;
    for (int img=-1; img<=1; ++img) {     // the window and its periodic images
      Real c = xc + img*jw_lx1;
      if ((x1max > c - jw_win_hw) && (x1min < c + jw_win_hw)) in = true;
    }
    refine_flag.h_view(m + mbs) = in ? 1 : -1;
  }
  refine_flag.modify_host();
  refine_flag.sync_device();
}

//----------------------------------------------------------------------------------------
//! \fn void JeansWaveErrors()
//! \brief Measures the Jeans wave growth rate (unstable) or oscillation
//!        frequency (stable)
//!        by Fourier-projecting the density perturbation onto the initial sin(k*x) mode.
//!
//! Computes A_sin = (2/V) * integral[ (rho/rho0 - 1) * sin(k*x) dV ], then:
//!   unstable (n_jeans > 1): omega_measured = ln(|A_sin| / amp) / t
//!   stable   (n_jeans < 1): omega_measured = arccos(A_sin / amp) / t
//! and compares against the analytical dispersion relation omega.

void JeansWaveErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  DvceArray5D<Real> u0;
  if (pmbp->phydro != nullptr) {
    u0 = pmbp->phydro->u0;
  } else if (pmbp->pmhd != nullptr) {
    u0 = pmbp->pmhd->u0;
  } else {
    return;
  }

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nk = ke - ks + 1, nj = je - js + 1, ni = ie - is + 1;
  int nmb = pmbp->nmb_thispack;
  int nmkji = nmb * nk * nj * ni;

  auto &size = pmbp->pmb->mb_size;
  Real t = pm->time;

  Real rho0 = jw_rho0;
  Real amp = jw_amp;
  Real k_wave = jw_k_wave;
  Real omega = jw_omega;
  Real n_jeans = jw_n_jeans;
  Real cos_a2 = jw_cos_a2;
  Real cos_a3 = jw_cos_a3;
  Real sin_a2 = jw_sin_a2;
  Real sin_a3 = jw_sin_a3;
  Real v0 = jw_v0;
  bool is_unstable = jw_unstable;

  Real sin_proj = 0.0;

  Kokkos::parallel_reduce(
      "jeans_wave_sin_proj",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, Real &sp) {
        int ii = idx % ni;
        int jj = (idx / ni) % nj;
        int kk = (idx / (ni * nj)) % nk;
        int mm = idx / (ni * nj * nk);
        int i = is + ii, j = js + jj, k = ks + kk;

        Real x1min = size.d_view(mm).x1min, x1max = size.d_view(mm).x1max;
        Real x2min = size.d_view(mm).x2min, x2max = size.d_view(mm).x2max;
        Real x3min = size.d_view(mm).x3min, x3max = size.d_view(mm).x3max;
        Real vol = size.d_view(mm).dx1 * size.d_view(mm).dx2 * size.d_view(mm).dx3;

        Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);
        Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);
        Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

        Real x = cos_a2*(x1v*cos_a3 + x2v*sin_a3) + x3v*sin_a2;
        sp += (u0(mm, IDN, k, j, i) / rho0 - 1.0)
              * std::sin(k_wave * (x - v0 * t)) * vol;
      },
      Kokkos::Sum<Real>(sin_proj));

#if MPI_PARALLEL_ENABLED
  Real global_sp = 0.0;
  MPI_Allreduce(&sin_proj, &global_sp, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sin_proj = global_sp;
#endif

  Real x1size = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real x2size = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real x3size = pm->mesh_size.x3max - pm->mesh_size.x3min;
  Real tvol = x1size * x2size * x3size;

  Real A_sin = 2.0 * sin_proj / tvol;
  Real omega_measured;
  if (is_unstable) {
    omega_measured = std::log(std::fabs(A_sin) / amp) / t;
  } else {
    Real clamped = fmax(-1.0, fmin(1.0, A_sin / amp));
    omega_measured = std::acos(clamped) / t;
  }

  if (global_variable::my_rank == 0) {
    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1);
    std::cout << "=====================================================" << std::endl;
    std::cout << "Jeans wave mode amplitude  : " << A_sin << std::endl;
    std::cout << "Jeans wave growth (A/amp)  : " << A_sin / amp << std::endl;
    std::cout << "Jeans wave omega measured  : " << omega_measured << std::endl;
    std::cout << "Jeans wave omega analytical: " << omega << std::endl;
    std::cout << "=====================================================" << std::endl;
  }

  // ---- Tomida & Stone (2023) sec. 4.3 error metric ------------------------------------
  // L1 errors of the conserved variables (rho, M1, M2, M3, E) against the analytic
  // linear solution, and their rms. The initial perturbation decomposes into a static
  // self-gravitating mode (fraction f_st; zero for an isentropic perturbation, and
  // (1-1/gamma)/(1-nu^2) for the isothermal perturbation of their setup, which has
  // delta_p = c_s^2 nu^2 delta_rho and no velocity) plus a standing gravito-acoustic
  // wave (fraction f_ac = 1-f_st, oscillating at the dispersion-relation omega):
  //   drho(x,t) = rho0 A sin(kx) [f_st + f_ac cos(omega t)]
  //   M(x,t)    = -k_hat rho0 A (omega/k) f_ac sin(omega t) cos(kx)
  //   p(x,t)    = p0 (1 + gamma A sin(kx) [nu^2 f_st + f_ac cos(omega t)])
  // Valid for the stable (nu < 1), v0 = 0, ideal-EOS case only.
  if (jw_ts_errors) {
    if (!jw_is_ideal || is_unstable || v0 != 0.0) {
      if (global_variable::my_rank == 0) {
        std::cout << "### WARNING: ts_errors requires the stable, v0=0, ideal-EOS "
                  << "case; skipping" << std::endl;
      }
      return;
    }
    Real p0 = jw_p0, gamma = jw_gamma;
    Real nu = n_jeans;
    Real f_st = jw_perturb_iso ? (1.0 - 1.0/gamma)/(1.0 - nu*nu) : 0.0;
    Real f_ac = 1.0 - f_st;
    Real gm1 = gamma - 1.0;
    Real cwt = std::cos(omega*t), swt = std::sin(omega*t);

    Real l1d = 0.0, l1m1 = 0.0, l1m2 = 0.0, l1m3 = 0.0, l1e = 0.0;
    Kokkos::parallel_reduce("jeans_ts43_errs",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &ed, Real &em1, Real &em2, Real &em3, Real &ee) {
      int ii = idx % ni;
      int jj = (idx / ni) % nj;
      int kk = (idx / (ni * nj)) % nk;
      int mm = idx / (ni * nj * nk);
      int i = is + ii, j = js + jj, k = ks + kk;
      Real x1min = size.d_view(mm).x1min, x1max = size.d_view(mm).x1max;
      Real x2min = size.d_view(mm).x2min, x2max = size.d_view(mm).x2max;
      Real x3min = size.d_view(mm).x3min, x3max = size.d_view(mm).x3max;
      Real vol = size.d_view(mm).dx1 * size.d_view(mm).dx2 * size.d_view(mm).dx3;
      Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);
      Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);
      Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);
      Real x = cos_a2*(x1v*cos_a3 + x2v*sin_a3) + x3v*sin_a2;
      Real sinkx = std::sin(k_wave*x), coskx = std::cos(k_wave*x);
      Real d_ana = rho0*(1.0 + amp*sinkx*(f_st + f_ac*cwt));
      Real mpar  = -rho0*amp*(omega/k_wave)*f_ac*swt*coskx;
      Real m1_ana = mpar*cos_a2*cos_a3;
      Real m2_ana = mpar*cos_a2*sin_a3;
      Real m3_ana = mpar*sin_a2;
      Real p_ana = p0*(1.0 + gamma*amp*sinkx*(nu*nu*f_st + f_ac*cwt));
      Real e_ana = p_ana/gm1 + 0.5*mpar*mpar/d_ana;
      ed  += Kokkos::fabs(u0(mm,IDN,k,j,i) - d_ana)*vol;
      em1 += Kokkos::fabs(u0(mm,IM1,k,j,i) - m1_ana)*vol;
      em2 += Kokkos::fabs(u0(mm,IM2,k,j,i) - m2_ana)*vol;
      em3 += Kokkos::fabs(u0(mm,IM3,k,j,i) - m3_ana)*vol;
      ee  += Kokkos::fabs(u0(mm,IEN,k,j,i) - e_ana)*vol;
    }, Kokkos::Sum<Real>(l1d), Kokkos::Sum<Real>(l1m1), Kokkos::Sum<Real>(l1m2),
       Kokkos::Sum<Real>(l1m3), Kokkos::Sum<Real>(l1e));
#if MPI_PARALLEL_ENABLED
    Real sums[5] = {l1d, l1m1, l1m2, l1m3, l1e};
    MPI_Allreduce(MPI_IN_PLACE, sums, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    l1d = sums[0]; l1m1 = sums[1]; l1m2 = sums[2]; l1m3 = sums[3]; l1e = sums[4];
#endif
    l1d /= tvol; l1m1 /= tvol; l1m2 /= tvol; l1m3 /= tvol; l1e /= tvol;
    Real rms = std::sqrt((SQR(l1d) + SQR(l1m1) + SQR(l1m2) + SQR(l1m3) + SQR(l1e))/5.0);
    if (global_variable::my_rank == 0) {
      std::cout << "# TS43-ERRS: rms_l1= " << rms
                << " l1_dens= " << l1d << " l1_mom1= " << l1m1
                << " l1_mom2= " << l1m2 << " l1_mom3= " << l1m3
                << " l1_etot= " << l1e << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void JeansWaveHistory()
//! \brief Projections of the solution onto the wave (see the file header): density
//! (a_sin, a_cos), perturbed field along e_perp (bp_sin), and velocity along k_hat and
//! e_perp (vpar_cos, vperp_cos), each (2/V) int (.) sin|cos(k xi) dV with xi = k_hat.x;
//! with MHD divb = int |div B| dx_min dV (face fields) and babs = int |B| dV, so that
//! divb/babs is the mean relative divergence; nmb = the number of MeshBlocks.

void JeansWaveHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const bool mhd = (pmbp->pmhd != nullptr);
  pdata->nhist = 8;
  pdata->label[0] = "a_sin";
  pdata->label[1] = "a_cos";
  pdata->label[2] = "bp_sin";
  pdata->label[3] = "vpar_cos";
  pdata->label[4] = "vperp_cos";
  pdata->label[5] = "divb";
  pdata->label[6] = "babs";
  pdata->label[7] = "nmb";
  auto &w0 = mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  DvceArray5D<Real> bcc;
  if (mhd) bcc = pmbp->pmhd->bcc0;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ni = indcs.nx1, nj = indcs.nx2, nk = indcs.nx3;
  int nmkji = pmbp->nmb_thispack*nk*nj*ni;
  auto &size = pmbp->pmb->mb_size;
  Real kw = jw_k_wave, rho0 = jw_rho0, bp0 = jw_bperp0;
  Real k1 = jw_khat[0], k2 = jw_khat[1], k3 = jw_khat[2];
  Real e1 = jw_eperp[0], e2 = jw_eperp[1], e3 = jw_eperp[2];
  Real vol_tot = (pm->mesh_size.x1max - pm->mesh_size.x1min)*
                 (pm->mesh_size.x2max - pm->mesh_size.x2min)*
                 (pm->mesh_size.x3max - pm->mesh_size.x3min);
  Real as = 0.0, ac = 0.0, bs = 0.0, vp = 0.0, ve = 0.0;
  Kokkos::parallel_reduce("jw_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(int idx, Real &s1, Real &s2, Real &s3, Real &s4, Real &s5) {
    int i = is + (idx % ni);
    int j = js + ((idx/ni) % nj);
    int k = ks + ((idx/(ni*nj)) % nk);
    int m = idx/(ni*nj*nk);
    Real x1v = CellCenterX(i-is, ni, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, nj, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k-ks, nk, size.d_view(m).x3min, size.d_view(m).x3max);
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real ph = kw*(k1*x1v + k2*x2v + k3*x3v);
    Real sn = sin(ph), cs = cos(ph);
    Real d = w0(m,IDN,k,j,i)/rho0 - 1.0;
    s1 += d*sn*vol;
    s2 += d*cs*vol;
    if (mhd) {
      Real bp = bcc(m,IBX,k,j,i)*e1 + bcc(m,IBY,k,j,i)*e2 + bcc(m,IBZ,k,j,i)*e3 - bp0;
      s3 += bp*sn*vol;
    }
    Real vx = w0(m,IVX,k,j,i), vy = w0(m,IVY,k,j,i), vz = w0(m,IVZ,k,j,i);
    s4 += (vx*k1 + vy*k2 + vz*k3)*cs*vol;
    s5 += (vx*e1 + vy*e2 + vz*e3)*cs*vol;
  }, Kokkos::Sum<Real>(as), Kokkos::Sum<Real>(ac), Kokkos::Sum<Real>(bs),
     Kokkos::Sum<Real>(vp), Kokkos::Sum<Real>(ve));
  // the history output sums the ranks
  pdata->hdata[0] = 2.0*as/vol_tot;
  pdata->hdata[1] = 2.0*ac/vol_tot;
  pdata->hdata[2] = 2.0*bs/vol_tot;
  pdata->hdata[3] = 2.0*vp/vol_tot;
  pdata->hdata[4] = 2.0*ve/vol_tot;
  Real db = 0.0, ba = 0.0;
  if (mhd) {
    auto &b0 = pmbp->pmhd->b0;
    Kokkos::parallel_reduce("jw_hist_divb", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &s1, Real &s2) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
      Real divb = (b0.x1f(m,k,j,i+1) - b0.x1f(m,k,j,i))/dx1 +
                  (b0.x2f(m,k,j+1,i) - b0.x2f(m,k,j,i))/dx2 +
                  (b0.x3f(m,k+1,j,i) - b0.x3f(m,k,j,i))/dx3;
      Real vol = dx1*dx2*dx3;
      s1 += vol*fabs(divb)*fmin(dx1, fmin(dx2, dx3));
      s2 += vol*sqrt(SQR(bcc(m,IBX,k,j,i)) + SQR(bcc(m,IBY,k,j,i)) + SQR(bcc(m,IBZ,k,j,i)));
    }, Kokkos::Sum<Real>(db), Kokkos::Sum<Real>(ba));
  }
  pdata->hdata[5] = db;
  pdata->hdata[6] = ba;
  pdata->hdata[7] = static_cast<Real>(pmbp->nmb_thispack);
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) pdata->hdata[n] = 0.0;
}
