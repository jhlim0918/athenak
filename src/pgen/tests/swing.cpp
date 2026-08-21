//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file swing.cpp
//! \brief Swing-amplification test for self-gravity in the shearing box.
//!
//! Initializes a leading shearing wave (kx0 < 0, ky > 0, kz = 0) of small
//! amplitude with zero velocity fluctuations:
//!     rho = rho0 [1 + A cos(kx0 x + ky y + phase0)
//!                   + A2 cos(kx2 x + ky2 y + phase2)],   v' = 0.
//! Background shear winds the wave up, kx(t) = kx0 + q*Omega*ky*t; as it swings through
//! kx = 0 self-gravity transiently amplifies it (Goldreich & Lynden-Bell 1965;
//! Julian & Toomre 1966; Toomre 1981).
//!
//! Initial-condition options (<problem> block; defaults reproduce the original test
//! bitwise):
//!  - default            : single wave, zero phase (phase0 = 0, amp2 = 0);
//!  - phase0 != 0        : the same wave translated by -phase0/ky in x2; the whole
//!                         solution (and the history projection, which carries phase0)
//!                         must translate rigidly — the translation-covariance test;
//!  - nwx2/nwy2/amp2/phase2 : an optional second leading wave. A non-parallel second
//!                         mode with generic phase2 leaves no inversion center on the
//!                         x1 = 0 axis, so the shear preserves no parity and collapse
//!                         sites are not symmetry-pinned — the parity-breaking test.
//!                         (IC-only; the history stays projected on the primary.)
//!
//! The user history output projects the solution onto the *instantaneous* shearing
//! wavevector k(t) = (kx(t), ky, 0), giving the linear-theory amplitudes directly:
//!     d_cos  = (2/V) Int (rho/rho0 - 1) cos(k.x) dV      -> delta
//!     d_sin  = (2/V) Int (rho/rho0 - 1) sin(k.x) dV      -> should remain ~0
//!     vx_sin = (2/V) Int vx sin(k.x) dV                  -> wx
//!     vy_sin = (2/V) Int vy sin(k.x) dV                  -> wy
//!     kx_ky  = kx(t)/ky
//! These obey the linearized shearing-sheet system (kz = 0, isothermal)
//!     delta' = -(kx wx + ky wy)
//!     wx'    =  2 Omega wy     + kx (cs^2 + 4 pi G rho0 / D) delta
//!     wy'    = -(2-q) Omega wx + ky (cs^2 + 4 pi G rho0 / D) delta
//! with D the discrete Laplacian eigenvalue (-> -k^2 in the continuum), which the
//! companion Julia notebook integrates for comparison.

#include <cmath>
#include <iostream>
#include <string>

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
#include "pgen/pgen.hpp"

// User-defined history function
void SwingHistory(HistoryData *pdata, Mesh *pm);
// User-defined refinement criterion: prescribed moving ring (same as the shwave
// pgen's; here it exercises AMR + self-gravity in the shearing box, Phase 2c)
void SwingMovingRingRefine(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \struct SwingTestVariables
//! \brief container for variables shared with the user-history function

namespace {
struct SwingTestVariables {
  Real kx0, ky, phase0, qshear, omega0, rho0, inv_vol;
};
SwingTestVariables swing_var;

// parameters for the prescribed moving-ring AMR criterion
struct SwingRingVariables {
  Real xc0, xamp, freq, hwidth;
};
SwingRingVariables swing_ring_var;
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::SwingAmplification()
//! \brief sets up a leading shearing wave for the swing-amplification test

void ProblemGenerator::SwingAmplification(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "swing test is hydro-only" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->phydro->psbox_u == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "swing test requires a <shearing_box> block in the input file"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // Prescribed moving-ring AMR criterion (Phase 2c: AMR + multigrid self-gravity in
  // the shearing box). Same criterion as the shwave pgen: refine every MeshBlock
  // whose x1 extent overlaps [xc(t)-hwidth, xc(t)+hwidth], xc(t) = xc0 +
  // xamp*sin(freq*t), derefine all others. Requires <mesh_refinement>
  // refinement=adaptive and an <amr_criterion> block with method=user.
  // Must be enrolled on both fresh starts and restarts.
  if (pin->GetOrAddBoolean("problem", "amr_moving_ring", false)) {
    swing_ring_var.xc0    = pin->GetOrAddReal("problem", "ring_xc0", 0.0);
    swing_ring_var.xamp   = pin->GetOrAddReal("problem", "ring_xamp", 0.0);
    swing_ring_var.freq   = pin->GetOrAddReal("problem", "ring_freq", 1.0);
    swing_ring_var.hwidth = pin->GetReal("problem", "ring_hwidth");
    user_ref_func = SwingMovingRingRefine;
  }

  // geometry and wave numbers (mode numbers are w.r.t. the whole mesh)
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min;
  Real ly = msize.x2max - msize.x2min;
  Real lz = msize.x3max - msize.x3min;
  int nwx = pin->GetOrAddInteger("problem", "nwx", -6);   // < 0 => leading wave
  int nwy = pin->GetOrAddInteger("problem", "nwy", 1);
  Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-4);
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);

  swing_var.kx0 = 2.0*M_PI*static_cast<Real>(nwx)/lx;
  swing_var.ky  = 2.0*M_PI*static_cast<Real>(nwy)/ly;
  // constant phase offset of the initial wave (translation-covariance tests: the
  // whole solution must shift by -phase0/ky in x2, incl. the history projection)
  swing_var.phase0 = pin->GetOrAddReal("problem", "phase0", 0.0);

  // optional SECOND leading wave (parity-breaking tests: a non-parallel mode with
  // generic phase leaves no inversion center on the x=0 axis, so the shear breaks
  // all parity and no clump site is symmetry-pinned). IC-only; the history
  // projection stays on the primary. Defaults are bitwise-inert.
  int nwx2 = pin->GetOrAddInteger("problem", "nwx2", 0);
  int nwy2 = pin->GetOrAddInteger("problem", "nwy2", 0);
  Real amp2 = pin->GetOrAddReal("problem", "amp2", 0.0);
  Real phase2 = pin->GetOrAddReal("problem", "phase2", 0.0);
  Real kx2 = 2.0*M_PI*static_cast<Real>(nwx2)/lx;
  Real ky2 = 2.0*M_PI*static_cast<Real>(nwy2)/ly;
  swing_var.qshear = pmbp->phydro->psbox_u->qshear;
  swing_var.omega0 = pmbp->phydro->psbox_u->omega0;
  swing_var.rho0 = rho0;
  swing_var.inv_vol = 1.0/(lx*ly*lz);
  user_hist_func = SwingHistory;

  // gravitational constant (same duplicated-setter pattern as the other gravity pgens)
  if (pmbp->pgrav != nullptr) {
    Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
    pmbp->pgrav->four_pi_G = four_pi_G;
    if (pmbp->pgrav->pmgd != nullptr) {
      pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);
    }
  }

  if (restart) return;

  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  Real p0 = pin->GetOrAddReal("problem", "p0", 1.0);
  // in non-FARGO mode (orbital_advection=false) the background shear flow
  // vy = -qshear*omega0*x1 must be included in the initial conditions
  bool orb_adv = pin->GetOrAddBoolean("shearing_box", "orbital_advection", true);
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  auto sv = swing_var;

  // leading shearing wave in density; velocity fluctuations zero
  par_for("swing_init", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real den = rho0*(1.0 + amp*cos(sv.kx0*x1v + sv.ky*x2v + sv.phase0)
                         + amp2*cos(kx2*x1v + ky2*x2v + phase2));
    Real vy0 = (orb_adv) ? 0.0 : -(sv.qshear)*(sv.omega0)*x1v;
    u0(m,IDN,k,j,i) = den;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = den*vy0;
    u0(m,IM3,k,j,i) = 0.0;
    if (eos.is_ideal) {
      u0(m,IEN,k,j,i) = p0/gm1 + 0.5*den*SQR(vy0);
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SwingHistory()
//! \brief projects the solution onto the instantaneous shearing wavevector

void SwingHistory(HistoryData *pdata, Mesh *pm) {
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;
  auto sv = swing_var;
  Real kx = sv.kx0 + (sv.qshear)*(sv.omega0)*(pm->time)*(sv.ky);

  pdata->nhist = 5;
  pdata->label[0] = "d_cos";
  pdata->label[1] = "d_sin";
  pdata->label[2] = "vx_sin";
  pdata->label[3] = "vy_sin";
  pdata->label[4] = "kx_ky";

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &w0_ = pm->pmb_pack->phydro->w0;
  array_sum::GlobalSum sum_this_mb;

  Kokkos::parallel_reduce("SwingHist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real phase = kx*x1v + sv.ky*x2v + sv.phase0;
    Real cs_ = cos(phase), sn_ = sin(phase);
    Real dd = w0_(m,IDN,k,j,i)/sv.rho0 - 1.0;

    array_sum::GlobalSum hvars;
    hvars.the_array[0] = vol*dd*cs_;
    hvars.the_array[1] = vol*dd*sn_;
    hvars.the_array[2] = vol*w0_(m,IVX,k,j,i)*sn_;
    hvars.the_array[3] = vol*w0_(m,IVY,k,j,i)*sn_;
    hvars.the_array[4] = vol*kx/sv.ky;   // vol-weighted so it survives the MPI sum
    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
  Kokkos::fence();

  // volume-average and normalize: (2/V) Int f cos(k.x) dV = amplitude of f
  for (int n=0; n<4; ++n) {
    pdata->hdata[n] = 2.0*sum_this_mb.the_array[n]*sv.inv_vol;
  }
  pdata->hdata[4] = sum_this_mb.the_array[4]*sv.inv_vol;
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SwingMovingRingRefine()
//! \brief User AMR criterion: refine MBs whose x1 extent overlaps a prescribed moving
//! interval [xc(t)-hwidth, xc(t)+hwidth], derefine all others. Criterion depends only
//! on x1, so it naturally flags complete x2-rings; the shearing-box ring-sync and
//! x1-boundary vetoes in MeshRefinement::CheckForRefinement enforce the rest of the
//! refinement policy. Runs on host data (MeshBlock bounds), no kernel needed.
//! Identical to the shwave pgen's MovingRingRefine.

void SwingMovingRingRefine(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  auto &size = pmbp->pmb->mb_size;
  auto &rv = swing_ring_var;
  Real xc = rv.xc0 + rv.xamp*std::sin(rv.freq*(pmbp->pmesh->time));

  for (int m=0; m<(pmbp->nmb_thispack); ++m) {
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    if ((x1max > (xc - rv.hwidth)) && (x1min < (xc + rv.hwidth))) {
      refine_flag.h_view(m + mbs) = 1;
    } else {
      refine_flag.h_view(m + mbs) = -1;
    }
  }
  // sync host array with device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  return;
}
