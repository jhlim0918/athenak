//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kos_disk.cpp
//! \brief Isothermal, vertically stratified, self-gravitating, magnetized galactic gas
//! disk of Kim, Ostriker & Stone (2002, ApJ 581, 1080; KOS02), sec. 3: the magneto-
//! hydrostatic equilibrium under gaseous self-gravity and a linear external (stellar)
//! gravity g_ext = -(g_*/H_*) z, with an azimuthal field of uniform plasma beta, and a
//! single axisymmetric density mode for measuring its stability (KOS02 Fig. 3b).
//!
//! Equilibrium (KOS02 eq. 16 in dimensional form): with the total pressure
//! c_s^2 rho (1 + 1/(2 beta)),
//!     (1 + 1/(2 beta)) c_s^2 d ln(rho)/dz = -(g_*/H_*) z - 4 pi G m(z),  dm/dz = rho,
//! integrated from the midplane density rho00, which is found by bisection so that
//! 2 m(infinity) = Sigma0 (<problem> sigma0). The strength of the external gravity is
//! s0 = 4 pi G rho_inf H_*/g_* (KOS02 eq. 10), rho_inf = pi G Sigma0^2/(2 c_s^2) the
//! midplane density of the purely self-gravitating unmagnetized disk; the input must set
//! <shearing_box> vertical_gravity_coef = g_*/H_* = 4 pi G rho_inf/s0 (checked here), so
//! that the external gravity does not change with the rotation rate. The scale height is
//! H = Sigma0/(2 rho00) (KOS02 eq. 14), H_inf = c_s^2/(pi G Sigma0) (eq. 15).
//!
//! Field (MHD): B = B_y(z) y_hat with B_y = (c_s^2 rho0(z)/beta)^(1/2), i.e. KOS02's
//! beta = c_s^2/v_A^2 (eq. 9; not the plasma beta 2P/B^2 = 2 beta) with v_A^2 = B^2/rho
//! here, so v_A is uniform (KOS02 sec. 2.2), and the magnetic pressure is
//! c_s^2 rho/(2 beta).
//! Perturbation: rho = rho0(z) [1 + amp cos(k_x x)], k_x = 2 pi nwx/L_x,
//! velocities zero; with k_y = 0 the problem stays axisymmetric (the y extent may be a
//! few cells).
//!
//! Random perturbations (KOS02 sec. 2.2, for the 3-D models): <problem> pert_file names a
//! raw float64 array delta[k][j][i] over the whole root mesh (nx3, nx2, nx1), written by
//! validation/run/kos02/gen_noise.py with <|delta_k|^2> ~ k^(-11/3) for 1 <= k L_x/2 pi
//! <= 32 and an rms of 1%; rho = rho0(z)[1 + delta + amp cos(k_x x)] (set amp = 0).
//! Frozen self-gravity (KOS02 footnote 8): with self-gravity off and <problem>
//! frozen_sg = true, the equilibrium's own vertical self-gravity,
//! g_sg(z) = -4 pi G m(|z|) sign(z),
//! is applied as a fixed user source term, so the non-self-gravitating models start in
//! the same equilibrium and only the evolution of the perturbations ignores self-gravity.
//!
//! History: sig_cos, sig_sin = (2/(L_x L_y)) Int rho {cos, sin}(k_x x) dV, the surface-
//! density amplitude of the mode (an x-uniform relaxation of the discrete equilibrium
//! projects to zero); mass; mx_sin = (2/(L_x L_y)) Int rho v_x sin(k_x x) dV; with MHD
//! also divb = Int |div B| dx_min dV and babs = Int |B| dV.

#include <algorithm>
#include <cstdio>
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
#include "shearing_box/shearing_box.hpp"
#include "pgen/pgen.hpp"

void KOSDiskHistory(HistoryData *pdata, Mesh *pm);

void KOSDiskUserSrcs(Mesh *pm, const Real bdt);
void KOSDiskFinalize(ParameterInput *pin, Mesh *pm);

namespace {
struct KOSVariables {
  Real kx, inv_area;
  bool frozen_sg = false;
  Real hz;                       // spacing of the g_sg table
  int ntab;
  DvceArray1D<Real> gsg;         // |g_sg|(|z|) = 4 pi G m(|z|)
};
KOSVariables kos_var;
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::KOSDisk()

void ProblemGenerator::KOSDisk(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const bool mhd = (pmbp->pmhd != nullptr);
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min, ly = msize.x2max - msize.x2min;
  int nwx = pin->GetOrAddInteger("problem", "nwx", 1);
  kos_var.kx = 2.0*M_PI*static_cast<Real>(nwx)/lx;
  kos_var.inv_area = 1.0/(lx*ly);
  user_hist_func = KOSDiskHistory;
  kos_var.frozen_sg = pin->GetOrAddBoolean("problem", "frozen_sg", false);
  if (kos_var.frozen_sg) {
    user_srcs = true;
    user_srcs_func = KOSDiskUserSrcs;
    pgen_final_func = KOSDiskFinalize;   // frees the table before Kokkos::finalize
  }

  // (the equilibrium, and on restarts the frozen self-gravity table, are rebuilt below)
  auto eos = mhd ? pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  if (eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "kos_disk requires an isothermal EOS" << std::endl;
    exit(EXIT_FAILURE);
  }
  const Real cs2 = SQR(eos.iso_cs);
  const Real four_pi_G = pin->GetReal("gravity", "four_pi_G");
  const Real piG = 0.25*four_pi_G;
  const Real sigma0 = pin->GetReal("problem", "sigma0");
  const Real s0 = pin->GetOrAddReal("problem", "s0", -1.0);      // <= 0: no external
  const Real beta = pin->GetOrAddReal("problem", "beta", -1.0);  // <= 0: no field
  const Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-3);
  const Real dfloor_ic = pin->GetOrAddReal("problem", "dfloor_ic", 1.0e-6);
  if (mhd && beta <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "kos_disk with <mhd> needs <problem> beta > 0" << std::endl;
    exit(EXIT_FAILURE);
  }
  const Real fmag = (mhd ? 1.0 + 0.5/beta : 1.0);
  const Real hinf = cs2/(piG*sigma0);
  const Real rhoinf = piG*SQR(sigma0)/(2.0*cs2);
  const Real gext = (s0 > 0.0) ? four_pi_G*rhoinf/s0 : 0.0;   // g_*/H_*
  const Real vg = pin->GetOrAddReal("shearing_box", "vertical_gravity_coef", -1.0);
  if (std::abs(vg - gext) > 1.0e-8*std::max(gext, 1.0e-30)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "kos_disk: <shearing_box> vertical_gravity_coef = " << vg
              << " but s0 = " << s0 << " requires g_*/H_* = " << gext << std::endl;
    exit(EXIT_FAILURE);
  }

  // ---- equilibrium by shooting on rho00 ----------------------------------------------
  Real zmax = std::max(std::abs(msize.x3min), std::abs(msize.x3max));
  const Real zinf = std::max(zmax, 40.0*hinf);
  const int nfine = 1 << 16;
  const Real hz = zinf/static_cast<Real>(nfine);
  auto integrate = [&](Real rho00, std::vector<Real> *tab) {
    Real lr = std::log(rho00), m = 0.0;
    auto f = [&](Real z, Real l, Real mm, Real &dl, Real &dm) {
      dl = -(gext*z + four_pi_G*mm)/(cs2*fmag);
      dm = std::exp(l);
    };
    if (tab) (*tab)[0] = rho00;
    for (int n=0; n<nfine; ++n) {
      Real z = n*hz, a1, b1, a2, b2, a3, b3, a4, b4;
      f(z, lr, m, a1, b1);
      f(z + 0.5*hz, lr + 0.5*hz*a1, m + 0.5*hz*b1, a2, b2);
      f(z + 0.5*hz, lr + 0.5*hz*a2, m + 0.5*hz*b2, a3, b3);
      f(z + hz, lr + hz*a3, m + hz*b3, a4, b4);
      lr += hz*(a1 + 2.0*a2 + 2.0*a3 + a4)/6.0;
      m += hz*(b1 + 2.0*b2 + 2.0*b3 + b4)/6.0;
      if (tab) (*tab)[n+1] = std::exp(lr);
    }
    return 2.0*m;
  };
  Real lo = 1.0e-6*rhoinf, hi = 1.0e6*rhoinf;
  for (int it=0; it<200; ++it) {
    Real mid = std::sqrt(lo*hi);
    if (integrate(mid, nullptr) > sigma0) hi = mid; else lo = mid;
    if (hi/lo - 1.0 < 1.0e-14) break;
  }
  const Real rho00 = std::sqrt(lo*hi);
  std::vector<Real> tab(nfine + 1);
  integrate(rho00, &tab);
  const Real hscale = sigma0/(2.0*rho00);
  if (global_variable::my_rank == 0) {
    std::cout << "kos_disk: Sigma0 = " << sigma0 << ", s0 = " << s0 << ", beta = "
              << (mhd ? beta : -1.0) << ", rho00 = " << rho00 << ", H = " << hscale
              << ", H_inf = " << hinf << ", H/H_inf = " << hscale/hinf
              << ", g*/H* = " << gext << std::endl;
  }
  if (kos_var.frozen_sg) {
    // m(z) from the same table (trapezoid), g_sg = 4 pi G m
    kos_var.hz = hz;
    kos_var.ntab = nfine;
    Kokkos::realloc(kos_var.gsg, nfine + 1);
    auto h = Kokkos::create_mirror_view(kos_var.gsg);
    Real m = 0.0;
    h(0) = 0.0;
    for (int n=0; n<nfine; ++n) {
      m += 0.5*(tab[n] + tab[n+1])*hz;
      h(n+1) = four_pi_G*m;
    }
    Kokkos::deep_copy(kos_var.gsg, h);
  }
  if (restart) return;
  DvceArray1D<Real> d_tab("kos_tab", nfine + 1);
  {
    auto h = Kokkos::create_mirror_view(d_tab);
    for (int n=0; n<=nfine; ++n) h(n) = std::max(tab[n], dfloor_ic*rho00);
    Kokkos::deep_copy(d_tab, h);
  }

  // ---- optional random perturbation field (whole root mesh, float64 [k][j][i]) --------
  std::string pert_file = pin->GetOrAddString("problem", "pert_file", "");
  const int gnx1 = pmy_mesh_->mesh_indcs.nx1, gnx2 = pmy_mesh_->mesh_indcs.nx2;
  const int gnx3 = pmy_mesh_->mesh_indcs.nx3;
  const bool use_pert = !pert_file.empty();
  DvceArray1D<Real> d_pert("kos_pert", use_pert ? gnx1*gnx2*gnx3 : 1);
  if (use_pert) {
    std::vector<double> buf(static_cast<size_t>(gnx1)*gnx2*gnx3);
    FILE *fp = std::fopen(pert_file.c_str(), "rb");
    size_t nread = fp ? std::fread(buf.data(), sizeof(double), buf.size(), fp) : 0;
    if (fp) std::fclose(fp);
    if (nread != buf.size()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "kos_disk: could not read " << buf.size()
                << " doubles from pert_file = " << pert_file << std::endl;
      exit(EXIT_FAILURE);
    }
    auto h = Kokkos::create_mirror_view(d_pert);
    for (size_t n=0; n<buf.size(); ++n) h(n) = static_cast<Real>(buf[n]);
    Kokkos::deep_copy(d_pert, h);
  }
  const Real gx1min = msize.x1min, gx2min = msize.x2min, gx3min = msize.x3min;
  const Real gdx1 = (msize.x1max - msize.x1min)/gnx1;
  const Real gdx2 = (msize.x2max - msize.x2min)/gnx2;
  const Real gdx3 = (msize.x3max - msize.x3min)/gnx3;

  // ---- fill the grid -----------------------------------------------------------------
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks;
  int ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  const Real kx = kos_var.kx;
  const bool oa = pin->GetOrAddBoolean("shearing_box", "orbital_advection", true);
  const Real qom = pin->GetReal("shearing_box", "qshear")*
                   pin->GetReal("shearing_box", "omega0");
  auto rho_at = [=] KOKKOS_FUNCTION (Real z) {
    Real rl = fabs(z)/hz;
    int l = static_cast<int>(rl);
    l = (l > nfine-1) ? nfine-1 : l;
    Real w = rl - static_cast<Real>(l);
    return (1.0-w)*d_tab(l) + w*d_tab(l+1);
  };
  par_for("kos_init", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real dpert = 0.0;
    if (use_pert) {
      Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      int gi = static_cast<int>((x1v - gx1min)/gdx1);
      int gj = static_cast<int>((x2v - gx2min)/gdx2);
      int gk = static_cast<int>((x3v - gx3min)/gdx3);
      dpert = d_pert((gk*gnx2 + gj)*gnx1 + gi);
    }
    Real den = rho_at(x3v)*(1.0 + dpert + amp*cos(kx*x1v));
    Real vy = oa ? 0.0 : -qom*x1v;
    u0(m,IDN,k,j,i) = den;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = den*vy;
    u0(m,IM3,k,j,i) = 0.0;
  });
  if (mhd) {
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    // KOS02 beta = c_s^2/v_A^2, v_A^2 = B^2/rho (Heaviside-Lorentz)
    const Real bfac = cs2/beta;
    par_for("kos_binit", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real by = sqrt(bfac*rho_at(x3v));
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = by;
      b0.x3f(m,k,j,i) = 0.0;
      if (i == ie) b0.x1f(m,k,j,i+1) = 0.0;
      if (j == je) b0.x2f(m,k,j+1,i) = by;
      if (k == ke) b0.x3f(m,k+1,j,i) = 0.0;
      bcc0(m,IBX,k,j,i) = 0.0;
      bcc0(m,IBY,k,j,i) = by;
      bcc0(m,IBZ,k,j,i) = 0.0;
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn KOSDiskHistory()
//! \brief surface-density and momentum amplitude of the axisymmetric mode

void KOSDiskHistory(HistoryData *pdata, Mesh *pm) {
  auto &size = pm->pmb_pack->pmb->mb_size;
  const bool mhd = (pm->pmb_pack->pmhd != nullptr);
  pdata->nhist = mhd ? 6 : 4;
  pdata->label[0] = "sig_cos";
  pdata->label[1] = "sig_sin";
  pdata->label[2] = "mass";
  pdata->label[3] = "mx_sin";
  if (mhd) {
    pdata->label[4] = "divb";
    pdata->label[5] = "babs";
  }
  int nh = pdata->nhist;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1, js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &w0_ = mhd ? pm->pmb_pack->pmhd->w0 : pm->pmb_pack->phydro->w0;
  DvceArray5D<Real> bcc_;
  DvceArray4D<Real> bx1f, bx2f, bx3f;
  if (mhd) {
    bcc_ = pm->pmb_pack->pmhd->bcc0;
    bx1f = pm->pmb_pack->pmhd->b0.x1f;
    bx2f = pm->pmb_pack->pmhd->b0.x2f;
    bx3f = pm->pmb_pack->pmhd->b0.x3f;
  }
  const Real kx = kos_var.kx;
  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("KOSHist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
    Real vol = dx1*dx2*dx3;
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real cs_ = cos(kx*x1v), sn_ = sin(kx*x1v);
    Real den = w0_(m,IDN,k,j,i);
    array_sum::GlobalSum h;
    h.the_array[0] = vol*den*cs_;
    h.the_array[1] = vol*den*sn_;
    h.the_array[2] = vol*den;
    h.the_array[3] = vol*den*w0_(m,IVX,k,j,i)*sn_;
    if (mhd) {
      Real divb = (bx1f(m,k,j,i+1) - bx1f(m,k,j,i))/dx1
                + (bx2f(m,k,j+1,i) - bx2f(m,k,j,i))/dx2
                + (bx3f(m,k+1,j,i) - bx3f(m,k,j,i))/dx3;
      h.the_array[4] = vol*fabs(divb)*fmin(dx1, fmin(dx2, dx3));
      h.the_array[5] = vol*sqrt(SQR(bcc_(m,IBX,k,j,i)) + SQR(bcc_(m,IBY,k,j,i))
                                + SQR(bcc_(m,IBZ,k,j,i)));
    }
    for (int n=nh; n<NHISTORY_VARIABLES; ++n) h.the_array[n] = 0.0;
    mb_sum += h;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));
  Kokkos::fence();
  pdata->hdata[0] = 2.0*sum.the_array[0]*kos_var.inv_area;
  pdata->hdata[1] = 2.0*sum.the_array[1]*kos_var.inv_area;
  pdata->hdata[2] = sum.the_array[2];
  pdata->hdata[3] = 2.0*sum.the_array[3]*kos_var.inv_area;
  if (mhd) {
    pdata->hdata[4] = sum.the_array[4];
    pdata->hdata[5] = sum.the_array[5];
  }
  for (int n=nh; n<NHISTORY_VARIABLES; ++n) pdata->hdata[n] = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn KOSDiskUserSrcs()
//! \brief the frozen equilibrium self-gravity g_sg(z) = -4 pi G m(|z|) sign(z), applied
//! after each stage's update (models without self-gravity, KOS02 footnote 8)

void KOSDiskUserSrcs(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const bool mhd = (pmbp->pmhd != nullptr);
  auto &u0 = mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks;
  int ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto gsg = kos_var.gsg;
  const Real hz = kos_var.hz;
  const int ntab = kos_var.ntab;
  const bool ideal = mhd ? pmbp->pmhd->peos->eos_data.is_ideal
                         : pmbp->phydro->peos->eos_data.is_ideal;
  par_for("kos_frozen_sg", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rl = fabs(z)/hz;
    int l = static_cast<int>(rl);
    l = (l > ntab-1) ? ntab-1 : l;
    Real w = rl - static_cast<Real>(l);
    Real g = -((1.0-w)*gsg(l) + w*gsg(l+1))*((z > 0.0) ? 1.0 : -1.0);
    Real den = u0(m,IDN,k,j,i);
    if (ideal) u0(m,IEN,k,j,i) += bdt*g*u0(m,IM3,k,j,i);
    u0(m,IM3,k,j,i) += bdt*den*g;
  });
}

//----------------------------------------------------------------------------------------
//! \fn KOSDiskFinalize()
//! \brief releases the device table held in a static (must precede Kokkos::finalize)

void KOSDiskFinalize(ParameterInput *pin, Mesh *pm) {
  kos_var.gsg = DvceArray1D<Real>();
}
