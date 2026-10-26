#ifndef SRCTERMS_SRCTERMS_HPP_
#define SRCTERMS_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.hpp
//! \brief Data, functions, and classes to implement various source terms in the hydro
//! and/or MHD equations of motion.  Currently implemented:
//!  (1) constant (gravitational) acceleration - for RTI
//!  (2) shearing box in 2D (x-z), for both hydro and MHD
//!  (3) random forcing to drive turbulence - implemented in TurbulenceDriver class

#include <map>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class SourceTerms
//! \brief data and functions for physical source terms

class SourceTerms {
 public:
  SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~SourceTerms();

  // data
  // flags for various source terms
  bool const_accel;
  bool ism_cooling;
  bool rel_cooling;
  bool beta_cooling;
  bool thermal_cooling;
  bool rad_beam;
  bool self_gravity;

  // new timestep
  Real dtnew;

  // data for constant accel
  Real const_accel_val;   // magnitude of accn
  int const_accel_dir;    // direction of accn

  // data for ISM cooling
  Real hrate;

  // data for relativistic cooling
  Real crate_rel;
  Real cpower_rel;

  // data for constant-beta cooling (Gammie 2001): rho*L = U/t_cool, t_cool = beta/Omega
  Real bcool_beta;      // beta = Omega*t_cool (the final beta when ramped)
  Real bcool_beta0;     // beta at t = 0 of a linear ramp (= bcool_beta: no ramp)
  Real bcool_t_ramp = 0.0;     // ramp time: beta reaches bcool_beta at t = bcool_t_ramp
  Real bcool_cs2_floor = 0.0;  // irradiation floor of beta cooling (cs^2 units)
  Real bcool_rho_cut = 0.0;    // first-core cutoff: cooling * 1/(1+(rho/rho_cut)^4), 0=off
  Real bcool_omega0;    // orbital frequency Omega

  // data for optically thin thermal cooling (Shi & Chiang 2014 eq. 8):
  // rho*L = U/t_cool with t_cool = b*(rho/P)^3 (constant opacity kappa)
  Real tcool_b;         // proportionality constant b

  // timestep safety factor for cooling source terms: dt <= cool_eps*t_cool
  Real cool_eps;

  // data for radiation beam source
  Real dii_dt;            // injection rate
  Real pos1, pos2, pos3;  // position of source
  Real dir1, dir2, dir3;  // direction of source
  Real width, spread;     // spatial width of source region, spread in angles

  // functions
  // beta of the beta cooling at time t: linear from bcool_beta0 at t = 0 to bcool_beta
  // at t = bcool_t_ramp, then constant
  Real BCoolBeta(const Real t) const {
    if (bcool_t_ramp <= 0.0 || t >= bcool_t_ramp) return bcool_beta;
    return bcool_beta0 + (bcool_beta - bcool_beta0)*fmax(t, 0.0)/bcool_t_ramp;
  }
  void ApplySrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, DvceArray5D<Real> &u0);
  void ApplySrcTerms(DvceArray5D<Real> &i0, const Real bdt);
  void ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, DvceArray5D<Real> &u0);
  void ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void BetaCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                   const Real bdt, DvceArray5D<Real> &u0);
  void ThermalCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                      const Real bdt, DvceArray5D<Real> &u0);
  void SelfGravity(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                   const Real bdt, DvceArray5D<Real> &u0);
  void BeamSource(DvceArray5D<Real> &i0, const Real bdt);
  void NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos);

 private:
  MeshBlockPack *pmy_pack;
};

#endif  // SRCTERMS_SRCTERMS_HPP_
