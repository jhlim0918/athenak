#ifndef GRAVITY_FFT_GRAVITY_HPP_
#define GRAVITY_FFT_GRAVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fft_gravity.hpp
//! \brief FFT-based Poisson solver for self-gravity on a uniform grid, including
//! shearing-box support via the phase-shift method (Gammie 2001) and open vertical
//! boundary conditions via the two-solve method (Koyama & Ostriker 2009).
//! Requires build with -D Athena_ENABLE_FFT=ON (kokkos-fft); this header is only
//! included from code compiled under FFT_ENABLED.

#include <memory>

#include <KokkosFFT.hpp>

#include "athena.hpp"

class MeshBlockPack;
class ParameterInput;
class Driver;

namespace gravity {

//! vertical (x3) boundary condition for the Poisson solve
enum class FFTVertBC {periodic, open};

//----------------------------------------------------------------------------------------
//! \class FFTGravitySolver
//! \brief solves 7-point discrete Poisson eqn on the (uniform) root grid with FFTs and
//! stores potential in Gravity::phi (active zones + one face-adjacent ghost layer).

class FFTGravitySolver {
 public:
  FFTGravitySolver(MeshBlockPack *pmbp, ParameterInput *pin);
  ~FFTGravitySolver() = default;

  // main interface, called once per RK stage (pdriver may be nullptr in pgen use)
  void Solve(Driver *pdriver, int stage);

  MeshBlockPack *pmy_pack;
  FFTVertBC vert_bc;                 // <gravity>/vert_bc = periodic (default) | open
  ReconstructionMethod remap_order;  // <gravity>/fft_remap = dc | plm (default) | ppmx
  bool shearing_box;                 // true if <shearing_box> block present
  Real qshear, omega0;               // shear parameters (0 if not shearing)

 private:
  using ComplexArray3D = DvceArray3D<Kokkos::complex<Real>>;
  using FFTPlan = KokkosFFT::Plan<DevExeSpace, ComplexArray3D, ComplexArray3D, 3>;

  int nx1_, nx2_, nx3_;              // root-grid cell counts
  Real dx1_, dx2_, dx3_;             // (uniform) cell sizes
  Real lx_, ly_, lz_;                // domain sizes
  Real x1min_;                       // inner x1 edge (for per-column shear offsets)

  DvceArray2D<int> goffs_;           // (nmb,3) global cell offset of each MeshBlock
  DvceArray3D<Real> rbuf_a_, rbuf_b_;  // (nx3,nx2,nx1) real work buffers
  ComplexArray3D zin_, zout_;          // (nx3,nx2,nx1) complex FFT work buffers
  DvceArray2D<Real> gplane_il_, gplane_iu_;  // (nx3,nx2) x1-face ghost planes

  std::unique_ptr<FFTPlan> plan_fwd_, plan_bwd_;

  // qshear*omega0*(time since last shear-periodic instant); 0 if not shearing
  Real ComputeQomt(Real time) const;
  // pack u0(IDN) active zones -> rbuf_a_, scaled by four_pi_G
  void GatherDensity(Real four_pi_G);
  // conservative per-x-column y-shift of content by (sgn_qomt * x1v) in dst = R[src];
  // roll: sgn_qomt = +qomt, unroll: sgn_qomt = -qomt (see notes in fft_gravity.cpp)
  void RollUnroll(const DvceArray3D<Real> &src, DvceArray3D<Real> &dst, Real sgn_qomt);
  // k-space solves: rbuf_b_ (rolled 4piG*rho) -> rbuf_a_ (rolled phi)
  void SolvePeriodic(Real qomt);
  void SolveOpen(Real qomt);
  // fill x1-face ghost planes from opposite-face columns of phi_g, y-shifted by
  // +/- qomt*lx (shear-periodic); plain periodic copy when qomt == 0
  void FillShearGhostPlanes(const DvceArray3D<Real> &phi_g, Real qomt);
  // phi_g -> pgrav->phi active zones + one face-adjacent ghost layer
  void ScatterPhi(const DvceArray3D<Real> &phi_g);
};

} // namespace gravity
#endif // GRAVITY_FFT_GRAVITY_HPP_
