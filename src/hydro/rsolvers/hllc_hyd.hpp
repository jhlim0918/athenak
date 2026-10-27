#ifndef HYDRO_RSOLVERS_HLLC_HYD_HPP_
#define HYDRO_RSOLVERS_HLLC_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hllc_hyd.hpp
//! \brief HLLC Riemann solver for hydrodynamics.  Reads L/R primitives from the global
//! per-face buffers and writes a single flux entry.  Ideal gas: HLLC of Toro et al.;
//! isothermal: HLLC_Isothermal (ported from athenak-dust amrSG_dev c6cdd22d, R. Li).

#include <cmath>

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn HLLC_Isothermal<ivx>()
//! \brief HLLC for isothermal hydrodynamics.  With p = cs^2 rho the middle wave cannot
//! carry a density jump; it is the shear (contact) wave of the transverse velocities.
//! This is the B = 0 limit of the isothermal HLLD solver (Mignone 2007, JCP 225, 1427):
//! density and normal momentum take the HLL flux, and the transverse momentum fluxes are
//! the HLL mass flux times the transverse velocity upwinded by the contact speed
//! S* = m_hll/rho_hll.  The outer wave speeds are those of HLLE (Einfeldt, Roe-averaged),
//! so the only difference from HLLE is the transverse-momentum flux: its dissipation is
//! |S*| rho dv_t/2 (as Roe) instead of the HLLE value ~ cs rho dv_t/2.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLC_Isothermal(const EOS_Data &eos,
                     const int m, const int k, const int j, const int i,
                     const DvceArray5D<Real> &wl,
                     const DvceArray5D<Real> &wr,
                     const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;
  const Real iso_cs = eos.iso_cs;
  const Real cs2 = iso_cs*iso_cs;

  Real dl = wl(m, IDN, k, j, i), ul = wl(m, ivx, k, j, i);
  Real vl = wl(m, ivy, k, j, i), zl = wl(m, ivz, k, j, i);
  Real dr = wr(m, IDN, k, j, i), ur = wr(m, ivx, k, j, i);
  Real vr = wr(m, ivy, k, j, i), zr = wr(m, ivz, k, j, i);

  // outer wave speeds as in HLLE (Roe-averaged normal velocity)
  Real sqrtdl = sqrt(dl), sqrtdr = sqrt(dr);
  Real uroe = (sqrtdl*ul + sqrtdr*ur)/(sqrtdl + sqrtdr);
  Real al = fmin(uroe - iso_cs, ul - iso_cs);
  Real ar = fmax(uroe + iso_cs, ur + iso_cs);

  // L/R fluxes of density and normal momentum
  Real fdl = dl*ul, fdr = dr*ur;
  Real fml = dl*ul*ul + cs2*dl, fmr = dr*ur*ur + cs2*dr;

  Real fd, fm;
  if (al >= 0.0) {
    fd = fdl; fm = fml;
  } else if (ar <= 0.0) {
    fd = fdr; fm = fmr;
  } else {
    Real idenom = 1.0/(ar - al);
    fd = (ar*fdl - al*fdr + al*ar*(dr - dl))*idenom;
    fm = (ar*fml - al*fmr + al*ar*(dr*ur - dl*ul))*idenom;
  }
  // contact speed S* = m_hll/rho_hll (only its sign is needed)
  Real dhll = (ar*dr - al*dl - (fdr - fdl))/(ar - al);
  Real mhll = (ar*dr*ur - al*dl*ul - (fmr - fml))/(ar - al);
  Real sstar = (al >= 0.0) ? ul : ((ar <= 0.0) ? ur : mhll/dhll);
  Real vt = (sstar > 0.0) ? vl : ((sstar < 0.0) ? vr : 0.5*(vl + vr));
  Real zt = (sstar > 0.0) ? zl : ((sstar < 0.0) ? zr : 0.5*(zl + zr));

  flx(m, IDN, k, j, i) = fd;
  flx(m, ivx, k, j, i) = fm;
  flx(m, ivy, k, j, i) = fd*vt;
  flx(m, ivz, k, j, i) = fd*zt;
}

//----------------------------------------------------------------------------------------
//! \fn HLLC<ivx>()
//! \brief Compute the HLLC flux at face (m,k,j,i) for direction ivx.  Isothermal EOS is
//! handled by HLLC_Isothermal above.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLC(const EOS_Data &eos,
          const int m, const int k, const int j, const int i,
          const int is, const int js, const int ks,
          const DvceArray5D<Real> &wl,
          const DvceArray5D<Real> &wr,
          const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;
  if (!eos.is_ideal) {
    HLLC_Isothermal<ivx>(eos, m, k, j, i, wl, wr, flx);
    return;
  }

  const Real gm1   = eos.gamma - 1.0;
  const Real igm1  = 1.0 / gm1;
  const Real alpha = ((eos.gamma) + 1.0) / (2.0 * (eos.gamma));


  // L/R primitives at face
  Real wl_idn = wl(m, IDN, k, j, i);
  Real wl_ivx = wl(m, ivx, k, j, i);
  Real wl_ivy = wl(m, ivy, k, j, i);
  Real wl_ivz = wl(m, ivz, k, j, i);

  Real wr_idn = wr(m, IDN, k, j, i);
  Real wr_ivx = wr(m, ivx, k, j, i);
  Real wr_ivy = wr(m, ivy, k, j, i);
  Real wr_ivz = wr(m, ivz, k, j, i);

  Real wl_ipr = eos.IdealGasPressure(wl(m, IEN, k, j, i));
  Real wr_ipr = eos.IdealGasPressure(wr(m, IEN, k, j, i));

  // PVRS middle-state estimates (Toro 10.5.2)
  Real qa, qb, qc, qd, qe, qf;
  qa = eos.IdealHydroSoundSpeed(wl_idn, wl_ipr);
  qb = eos.IdealHydroSoundSpeed(wr_idn, wr_ipr);
  Real el = wl_ipr * igm1 + 0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
  Real er = wr_ipr * igm1 + 0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
  qc = 0.25 * (wl_idn + wr_idn) * (qa + qb);
  qd = 0.5 * (wl_ipr + wr_ipr + (wl_ivx - wr_ivx) * qc);  // P_mid

  qe = (qd <= wl_ipr) ? 1.0 : sqrt(1.0 + alpha * ((qd / wl_ipr) - 1.0));
  qf = (qd <= wr_ipr) ? 1.0 : sqrt(1.0 + alpha * ((qd / wr_ipr) - 1.0));

  // Max/min wave speeds
  qc = wl_ivx - qa * qe;  // al
  qd = wr_ivx + qb * qf;  // ar

  qa = qd > 0.0 ? qd : 1.0e-20;   // bp
  qb = qc < 0.0 ? qc : -1.0e-20;  // bm

  // Contact wave speed and pressure
  qe = wl_ivx - qc;  // vxl
  qf = wr_ivx - qd;  // vxr

  qc = wl_ipr + qe * wl_idn * wl_ivx;  // tl
  qd = wr_ipr + qf * wr_idn * wr_ivx;  // tr

  Real ml =  wl_idn * qe;
  Real mr = -(wr_idn * qf);

  Real am = (qc - qd) / (ml + mr);
  Real cp = (ml * qd + mr * qc) / (ml + mr);
  cp = cp > 0.0 ? cp : 0.0;

  // L/R fluxes along bm (qb) and bp (qa)
  qe = wl_idn * (wl_ivx - qb);
  qf = wr_idn * (wr_ivx - qa);

  Real fl_d  = qe;
  Real fr_d  = qf;
  Real fl_mx = qe * wl_ivx + wl_ipr;
  Real fr_mx = qf * wr_ivx + wr_ipr;
  Real fl_my = qe * wl_ivy;
  Real fr_my = qf * wr_ivy;
  Real fl_mz = qe * wl_ivz;
  Real fr_mz = qf * wr_ivz;
  Real fl_e  = el * (wl_ivx - qb) + wl_ipr * wl_ivx;
  Real fr_e  = er * (wr_ivx - qa) + wr_ipr * wr_ivx;

  // Flux weights
  if (am >= 0.0) {
    qc =  am / (am - qb);
    qd = 0.0;
    qe = -qb / (am - qb);
  } else {
    qc =  0.0;
    qd = -am / (qa - am);
    qe =  qa / (qa - am);
  }

  // HLLC flux at interface
  flx(m, IDN, k, j, i) = qc * fl_d  + qd * fr_d;
  flx(m, ivx, k, j, i) = qc * fl_mx + qd * fr_mx + qe * cp;
  flx(m, ivy, k, j, i) = qc * fl_my + qd * fr_my;
  flx(m, ivz, k, j, i) = qc * fl_mz + qd * fr_mz;
  flx(m, IEN, k, j, i) = qc * fl_e  + qd * fr_e  + qe * cp * am;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLC_HYD_HPP_
