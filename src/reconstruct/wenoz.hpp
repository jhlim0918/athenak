#ifndef RECONSTRUCT_WENOZ_HPP_
#define RECONSTRUCT_WENOZ_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file wenoz.hpp
//! \brief WENO-Z reconstruction for a Cartesian-like coordinate with uniform spacing.
//!
//! REFERENCES:
//! Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially
//! non-oscillatory scheme for hyperbolic conservation laws" , JCP, 227, 3191 (2008)
//!
//! Castro M., Costa B., Don W.W. , "High order weighted essentially non-oscillatory
//! WENO-Z schemes for hyperbolic conservation laws" , JCP, 230, 1766 (2011)

#include <math.h>
#include <algorithm>    // max()

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn WENOZ()
//! \brief Reconstructs 5th-order polynomial in cell i to compute ql(i+1) and qr(i).
//! Works for any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void WENOZ(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
           const Real &q_ip2, Real &ql_ip1, Real &qr_i) noexcept  {
  // Smooth WENO weights: See Jiang & Shu 1996
  const Real beta_coeff[2]{13. / 12., 0.25};

  Real beta[3];
  beta[0] = beta_coeff[0] * SQR(q_im2 +     q_i - 2.0*q_im1) +
            beta_coeff[1] * SQR(q_im2 + 3.0*q_i - 4.0*q_im1);

  beta[1] = beta_coeff[0] * SQR(q_im1 + q_ip1 - 2.0*q_i) +
            beta_coeff[1] * SQR(q_im1 - q_ip1);

  beta[2] = beta_coeff[0] * SQR(q_ip2 +      q_i - 2.0*q_ip1) +
            beta_coeff[1] * SQR(q_ip2 + 3.0* q_i - 4.0*q_ip1);

  // Rescale epsilon
  const Real epsL = 1.0e-42;

  // WENO-Z: Borges et al. 2008, Castro et al. 2011
  const Real tau_5 = fabs(beta[0] - beta[2]);

  Real indicator[3];
  indicator[0] = SQR(tau_5 / (beta[0] + epsL));
  indicator[1] = SQR(tau_5 / (beta[1] + epsL));
  indicator[2] = SQR(tau_5 / (beta[2] + epsL));

  // compute qL_ip1
  // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
  Real f[3];
  f[0] = ( 2.0*q_im2 - 7.0*q_im1 + 11.0*q_i  );
  f[1] = (-1.0*q_im1 + 5.0*q_i   + 2.0 *q_ip1);
  f[2] = ( 2.0*q_i   + 5.0*q_ip1 -      q_ip2);

  Real alpha[3];
  alpha[0] = 0.1*(1.0 + indicator[0]);
  alpha[1] = 0.6*(1.0 + indicator[1]);
  alpha[2] = 0.3*(1.0 + indicator[2]);
  Real alpha_sum = 6.0*(alpha[0] + alpha[1] + alpha[2]);

  ql_ip1 = (f[0]*alpha[0] + f[1]*alpha[1] + f[2]*alpha[2])/alpha_sum;

  // compute qR_i
  // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
  f[0] = ( 2.0*q_ip2 - 7.0*q_ip1 + 11.0*q_i  );
  f[1] = (-1.0*q_ip1 + 5.0*q_i   + 2.0 *q_im1);
  f[2] = ( 2.0*q_i   + 5.0*q_im1 -      q_im2);

  alpha[0] = 0.1*(1.0 + indicator[2]);
  alpha[2] = 0.3*(1.0 + indicator[0]);
  alpha_sum = 6.0*(alpha[0] + alpha[1] + alpha[2]);

  qr_i = (f[0]*alpha[0] + f[1]*alpha[1] + f[2]*alpha[2])/alpha_sum;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZ_MC()
//! \brief WENO-Z face values constrained by the monotonized-central (MC) limiter.
//! MC limits the slope as minmod(2 dq_-, 2 dq_+, s), with s the centred slope; here s is
//! replaced by the one-sided slope WENO-Z implies at each face, 2 (q_face - q_i), so each
//! face deviation becomes minmod(q_face - q_i, dq_-, dq_+).  In smooth monotone regions
//! the WENO-Z value passes unchanged (|q_face - q_i| ~ |dq|/2); at extrema (dq_- dq_+ <= 0)
//! both faces drop to q_i, as with MC.

KOKKOS_INLINE_FUNCTION
Real MinMod3(const Real a, const Real b, const Real c) noexcept {
  if (a > 0.0 && b > 0.0 && c > 0.0) return fmin(a, fmin(b, c));
  if (a < 0.0 && b < 0.0 && c < 0.0) return fmax(a, fmax(b, c));
  return 0.0;
}

KOKKOS_INLINE_FUNCTION
void WENOZ_MC(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
              const Real &q_ip2, Real &ql_ip1, Real &qr_i) noexcept  {
  WENOZ(q_im2, q_im1, q_i, q_ip1, q_ip2, ql_ip1, qr_i);
  const Real dqm = q_i - q_im1;
  const Real dqp = q_ip1 - q_i;
  const Real dr = ql_ip1 - q_i;   // right-face deviation (half slope, +x sense)
  const Real dl = q_i - qr_i;     // left-face deviation (half slope, +x sense)
  ql_ip1 = q_i + MinMod3(dr, dqm, dqp);
  qr_i   = q_i - MinMod3(dl, dqm, dqp);
  return;
}
#endif // RECONSTRUCT_WENOZ_HPP_
