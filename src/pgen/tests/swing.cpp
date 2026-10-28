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
//!
//! MHD (an <mhd> block instead of <hydro>; 2026-10-26): a uniform azimuthal field
//! B = <problem>/b0 y_hat (Kim & Ostriker 2001 geometry, kz = 0), and optionally a uniform
//! radial field <problem>/bx0, which the shear winds up exactly, By = b0 - q Omega bx0 t (a
//! kinematic check of the field's shear term; amp = 0 for the pure test). The perturbation is
//! isentropic for an ideal-gas EOS (dp = cs^2 drho, cs^2 = gamma p0/rho0). The history adds
//!     bx_cos, by_cos = (2/V) Int dB_{x,y} cos(k.x) dV   -> cx, cy
//!     divb = Int |div B| dx_min dV,  babs = Int |B| dV,  nmb = number of MeshBlocks
//! and the linear system gains the field (Heaviside-Lorentz units, vA = B0/sqrt(rho0)):
//!     wx' += -(B0/rho0)(ky cx - kx cy),   cx' = ky B0 wx,   cy' = -kx B0 wx - q Omega cx
//! (validation/run/swing_mhd/swing_ode.py). MHD with orbital advection cannot refine, so
//! refined MHD runs use orbital_advection = false (the full shear flow in the state).

#include <algorithm>
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
  Real kx0, ky, phase0, qshear, omega0, rho0, inv_vol, b0;
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
  const bool mhd = (pmbp->pmhd != nullptr);
  ShearingBoxCC *psbox = mhd ? pmbp->pmhd->psbox_u : pmbp->phydro->psbox_u;
  if (psbox == nullptr) {
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
  swing_var.qshear = psbox->qshear;
  swing_var.omega0 = psbox->omega0;
  swing_var.rho0 = rho0;
  swing_var.b0 = pin->GetOrAddReal("problem", "b0", 0.0);   // MHD: uniform B along x2
  // MHD: optional uniform radial field (the shear winds it into By = b0 - q Omega bx0 t)
  Real bx0 = pin->GetOrAddReal("problem", "bx0", 0.0);
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

  EOS_Data &eos = mhd ? pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
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
  auto &u0 = mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto sv = swing_var;
  // MHD: isentropic perturbation (dp = cs^2 drho) and the field's energy; the hydro path
  // keeps its uniform pressure (bitwise as before)
  Real pfac = mhd ? eos.gamma : 0.0;
  Real eb = mhd ? 0.5*(SQR(sv.b0) + SQR(bx0)) : 0.0;

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
      Real dfrac = amp*cos(sv.kx0*x1v + sv.ky*x2v + sv.phase0)
                 + amp2*cos(kx2*x1v + ky2*x2v + phase2);
      u0(m,IEN,k,j,i) = p0*(1.0 + pfac*dfrac)/gm1 + 0.5*den*SQR(vy0) + eb;
    }
  });
  if (mhd) {
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    Real by0 = sv.b0;
    Real bxu = bx0;
    par_for("swing_bfield", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is,
    ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = bxu;
      b0.x2f(m,k,j,i) = by0;
      b0.x3f(m,k,j,i) = 0.0;
      if (i == ie) b0.x1f(m,k,j,i+1) = bxu;
      if (j == je) b0.x2f(m,k,j+1,i) = by0;
      if (k == ke) b0.x3f(m,k+1,j,i) = 0.0;
      bcc0(m,IBX,k,j,i) = bxu;
      bcc0(m,IBY,k,j,i) = by0;
      bcc0(m,IBZ,k,j,i) = 0.0;
    });
  }
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

  const bool mhd = (pm->pmb_pack->pmhd != nullptr);
  pdata->nhist = mhd ? 12 : 5;
  pdata->label[0] = "d_cos";
  pdata->label[1] = "d_sin";
  pdata->label[2] = "vx_sin";
  pdata->label[3] = "vy_sin";
  pdata->label[4] = "kx_ky";
  if (mhd) {
    pdata->label[5] = "bx_cos";
    pdata->label[6] = "by_cos";
    pdata->label[7] = "divb";
    pdata->label[8] = "babs";
    pdata->label[9] = "nmb";
    pdata->label[10] = "bmis";   // Sum |Bn(coarse copy) - mean Bn(fine copies)| dA
    pdata->label[11] = "bnrm";   // Sum |Bn| dA over the same level-jump faces
  }

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &w0_ = mhd ? pm->pmb_pack->pmhd->w0 : pm->pmb_pack->phydro->w0;
  DvceArray5D<Real> bcc_;
  DvceFaceFld4D<Real> *pb0 = nullptr;
  if (mhd) {bcc_ = pm->pmb_pack->pmhd->bcc0; pb0 = &(pm->pmb_pack->pmhd->b0);}
  DvceArray4D<Real> bx1f, bx2f, bx3f;
  if (mhd) {bx1f = pb0->x1f; bx2f = pb0->x2f; bx3f = pb0->x3f;}
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
    if (mhd) {
      Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
      hvars.the_array[5] = vol*bcc_(m,IBX,k,j,i)*cs_;
      hvars.the_array[6] = vol*(bcc_(m,IBY,k,j,i) - sv.b0)*cs_;
      Real divb = (bx1f(m,k,j,i+1) - bx1f(m,k,j,i))/dx1 + (bx2f(m,k,j+1,i) - bx2f(m,k,j,i))/dx2
                + (bx3f(m,k+1,j,i) - bx3f(m,k,j,i))/dx3;
      hvars.the_array[7] = vol*fabs(divb)*fmin(dx1, fmin(dx2, dx3));
      hvars.the_array[8] = vol*sqrt(SQR(bcc_(m,IBX,k,j,i)) + SQR(bcc_(m,IBY,k,j,i))
                                    + SQR(bcc_(m,IBZ,k,j,i)));
    }
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
  if (mhd) {
    pdata->hdata[5] = 2.0*sum_this_mb.the_array[5]*sv.inv_vol;
    pdata->hdata[6] = 2.0*sum_this_mb.the_array[6]*sv.inv_vol;
    pdata->hdata[7] = sum_this_mb.the_array[7];     // divb, babs: integrals (ratio)
    pdata->hdata[8] = sum_this_mb.the_array[8];
    pdata->hdata[9] = static_cast<Real>(pm->pmb_pack->nmb_thispack);   // ranks summed
    // Consistency of the two copies of a face field shared by a fine and a coarse
    // MeshBlock. After the boundary exchange a fine block holds, in coarse_b0, the coarse
    // neighbour's copy of the shared face; compare it with the mean of the fine block's
    // own 2x2 faces, on all six faces with a coarser neighbour.
    pdata->hdata[10] = 0.0;
    pdata->hdata[11] = 0.0;
    if (pm->multilevel) {
      auto &cb = pm->pmb_pack->pmhd->coarse_b0;
      auto &nghbr = pm->pmb_pack->pmb->nghbr;
      auto &mblev = pm->pmb_pack->pmb->mb_lev;
      int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
      int cis = indcs.cis, cie = indcs.cie, cjs = indcs.cjs, cje = indcs.cje;
      int cks = indcs.cks, cke = indcs.cke;
      int cnx1 = nx1/2, cnx2 = nx2/2, cnx3 = nx3/2;
      int nfe = std::max(cnx2*cnx3, std::max(cnx1*cnx3, cnx1*cnx2));
      int nm = pm->pmb_pack->nmb_thispack;
      Real mis = 0.0, nrm = 0.0;
      Kokkos::parallel_reduce("SwingBmis", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                              nm*6*nfe),
      KOKKOS_LAMBDA(const int &idx, Real &smis, Real &snrm) {
        int m = idx/(6*nfe);
        int f = (idx - m*6*nfe)/nfe;          // 0,1: -/+x1  2,3: -/+x2  4,5: -/+x3
        int e = idx - m*6*nfe - f*nfe;
        const int base[6] = {0, 4, 8, 12, 24, 28};
        bool coarser = false;
        for (int s=0; s<4; ++s) {
          int n = base[f] + s;
          if (nghbr.d_view(m,n).gid >= 0 && nghbr.d_view(m,n).lev < mblev.d_view(m)) {
            coarser = true;
          }
        }
        if (!coarser) return;
        Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
        Real bc, bf, da;
        if (f < 2) {
          if (e >= cnx2*cnx3) return;
          int ck = e/cnx2 + cks, cj = e%cnx2 + cjs;
          int ci = (f == 0) ? cis : cie+1, fi = (f == 0) ? is : ie+1;
          int fk = ks + 2*(ck-cks), fj = js + 2*(cj-cjs);
          bc = cb.x1f(m,ck,cj,ci);
          bf = 0.25*(bx1f(m,fk,fj,fi) + bx1f(m,fk,fj+1,fi) + bx1f(m,fk+1,fj,fi)
                   + bx1f(m,fk+1,fj+1,fi));
          da = 4.0*dx2*dx3;
        } else if (f < 4) {
          if (e >= cnx1*cnx3) return;
          int ck = e/cnx1 + cks, ci = e%cnx1 + cis;
          int cj = (f == 2) ? cjs : cje+1, fj = (f == 2) ? js : je+1;
          int fk = ks + 2*(ck-cks), fi = is + 2*(ci-cis);
          bc = cb.x2f(m,ck,cj,ci);
          bf = 0.25*(bx2f(m,fk,fj,fi) + bx2f(m,fk,fj,fi+1) + bx2f(m,fk+1,fj,fi)
                   + bx2f(m,fk+1,fj,fi+1));
          da = 4.0*dx1*dx3;
        } else {
          if (e >= cnx1*cnx2) return;
          int cj = e/cnx1 + cjs, ci = e%cnx1 + cis;
          int ck = (f == 4) ? cks : cke+1, fk = (f == 4) ? ks : ke+1;
          int fj = js + 2*(cj-cjs), fi = is + 2*(ci-cis);
          bc = cb.x3f(m,ck,cj,ci);
          bf = 0.25*(bx3f(m,fk,fj,fi) + bx3f(m,fk,fj,fi+1) + bx3f(m,fk,fj+1,fi)
                   + bx3f(m,fk,fj+1,fi+1));
          da = 4.0*dx1*dx2;
        }
        smis += fabs(bc - bf)*da;
        snrm += fabs(bf)*da;
      }, Kokkos::Sum<Real>(mis), Kokkos::Sum<Real>(nrm));
      pdata->hdata[10] = mis;
      pdata->hdata[11] = nrm;
    }
  }
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
  refine_flag.modify_host();
  refine_flag.sync_device();
  return;
}
