#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.hpp
//! \brief defines MGGravity class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../parameter_input.hpp"
#include "../multigrid/multigrid.hpp"
#include "../coordinates/coordinates.hpp"
#include "mg_gravity.hpp"

class MeshBlockPack;
class ParameterInput;
class Coordinates;
class Multigrid;
class Driver;
namespace gravity {
class FFTGravitySolver;
class Gravity {
 public:
  Gravity(MeshBlockPack *pmbp, ParameterInput *pin);
  ~Gravity();

  // dispatches to the solver selected by <gravity> solver = multigrid (default) | fft
  void Solve(Driver *pdriver, int stage);

  // An additional mass density (e.g. the particle-mesh dust density, dust track Phase
  // 4c) added to the gas density in every solver's source read.  The registering module
  // owns the array and keeps its active cells and one ghost layer valid before each
  // solve.  Solvers read SourceArray()(m,SourceIndex(),k,j,i): the gas conserved array
  // at IDN when nothing is registered, else the sum formed in Solve().
  void RegisterExtraDensity(const DvceArray5D<Real> &rho);
  const DvceArray5D<Real>& SourceArray() const;
  int SourceIndex() const;
  bool has_extra_density = false;

  MeshBlockPack* pmy_pack;
  DvceArray5D<Real> phi, coarse_phi;
  DvceArray5D<Real> def;
  DvceArray5D<Real> rho_extra;   // registered additional density (view alias)
  DvceArray5D<Real> rho_total;   // gas + extra, formed in Solve() when registered
  Real four_pi_G;
  bool output_defect;
  bool fill_ghost;
  MGGravityDriver *pmgd;
  MGGravity *pmg;
  FFTGravitySolver *pfft;
  void SaveFaceBoundaries();
  void RestoreFaceBoundaries();

  friend class MGGravityDriver;

 private:
  DvceArray5D<Real> fbuf_[6];
};
}  // namespace gravity
#endif // GRAVITY_GRAVITY_HPP_
