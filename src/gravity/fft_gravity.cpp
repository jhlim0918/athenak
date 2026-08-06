//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fft_gravity.cpp
//! \brief Implementation of FFT-based Poisson solver for self-gravity, following the
//! Athena-C solver in selfg_fft_disk.c:
//!  - shearing box handled by the phase-shift method (Gammie 2001): density is "rolled"
//!    (y-shifted by +q*Omega*t_s*x per x-column, t_s = time since last shear-periodic
//!    instant) into a strictly periodic frame, solved with the shear-shifted radial
//!    wavenumber kx' = kx + q*Omega*t_s*ky, and the potential "unrolled" back;
//!  - open (vacuum) vertical BC via the two-solve method (Koyama & Ostriker 2009):
//!    periodic solve A and half-integer-kz solve B combined with screening weights
//!    (1 -/+ exp(-kperp*Lz))/2, which cancels all vertical periodic images exactly.
//! The non-shearing case is the same code path with qomt = 0 (remap becomes identity).

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "gravity/gravity.hpp"
#include "gravity/fft_gravity.hpp"
#include "shearing_box/remap_fluxes.hpp"

namespace gravity {

// halo width for the conservative y-remap scratch arrays; PPMX needs j-2..j+2 plus the
// face at ju+1, so 3 is exactly sufficient (integer part of the shift is folded into
// the periodic-wrapped scratch load, so the halo does not depend on the shift size)
namespace {
constexpr int PAD = 3;
}

//----------------------------------------------------------------------------------------
//! \fn FFTGravitySolver::FFTGravitySolver()
//! \brief constructor: checks v1 restrictions, reads options, allocates global work
//! arrays sized to the root grid, and creates reusable FFT plans

FFTGravitySolver::FFTGravitySolver(MeshBlockPack *pmbp, ParameterInput *pin) :
    pmy_pack(pmbp) {
  Mesh *pm = pmbp->pmesh;

  // v1 restrictions
  if (global_variable::nranks > 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "FFT gravity solver currently supports single MPI rank only"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pm->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "FFT gravity solver requires a uniform grid (no SMR/AMR)" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pm->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "FFT gravity solver requires a 3D mesh" << std::endl;
    exit(EXIT_FAILURE);
  }

  // options
  std::string vbc = pin->GetOrAddString("gravity", "vert_bc", "periodic");
  if (vbc == "periodic") {
    vert_bc = FFTVertBC::periodic;
  } else if (vbc == "open") {
    vert_bc = FFTVertBC::open;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<gravity> vert_bc = '" << vbc << "' not recognized "
              << "(must be 'periodic' or 'open')" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string rmap = pin->GetOrAddString("gravity", "fft_remap", "plm");
  if (rmap == "dc") {
    remap_order = ReconstructionMethod::dc;
  } else if (rmap == "plm") {
    remap_order = ReconstructionMethod::plm;
  } else if (rmap == "ppmx") {
    remap_order = ReconstructionMethod::ppmx;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<gravity> fft_remap = '" << rmap << "' not recognized "
              << "(must be 'dc', 'plm', or 'ppmx')" << std::endl;
    exit(EXIT_FAILURE);
  }

  // shear parameters (same per-module read idiom as ShearingBox/OrbitalAdvection)
  shearing_box = pin->DoesBlockExist("shearing_box");
  qshear = 0.0;
  omega0 = 0.0;
  if (shearing_box) {
    qshear = pin->GetReal("shearing_box", "qshear");
    omega0 = pin->GetReal("shearing_box", "omega0");
  }

  // boundary-condition consistency: x1/x2 must be (shear-)periodic; x3 periodic unless
  // vert_bc = open
  {
    auto ok_x1 = [](BoundaryFlag f) {
      return (f == BoundaryFlag::periodic || f == BoundaryFlag::shear_periodic);
    };
    bool bc_ok = ok_x1(pm->mesh_bcs[BoundaryFace::inner_x1]) &&
                 ok_x1(pm->mesh_bcs[BoundaryFace::outer_x1]) &&
                 (pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic) &&
                 (pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic);
    if (vert_bc == FFTVertBC::periodic) {
      bc_ok = bc_ok && (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic)
                    && (pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic);
    }
    if (!bc_ok) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "FFT gravity solver requires (shear-)periodic x1, periodic x2, and "
                << "periodic x3 boundaries (unless <gravity> vert_bc = open)"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // root-grid geometry
  auto &mindcs = pm->mesh_indcs;
  nx1_ = mindcs.nx1;
  nx2_ = mindcs.nx2;
  nx3_ = mindcs.nx3;
  auto &msize = pm->mesh_size;
  lx_ = msize.x1max - msize.x1min;
  ly_ = msize.x2max - msize.x2min;
  lz_ = msize.x3max - msize.x3min;
  dx1_ = lx_/static_cast<Real>(nx1_);
  dx2_ = ly_/static_cast<Real>(nx2_);
  dx3_ = lz_/static_cast<Real>(nx3_);
  x1min_ = msize.x1min;

  // global cell-index offset of each MeshBlock in this pack (uniform grid: all blocks
  // at root level, so offset = logical location * block size)
  int nmb = pmbp->nmb_thispack;
  Kokkos::realloc(goffs_, nmb, 3);
  auto goffs_h = Kokkos::create_mirror_view(goffs_);
  for (int m=0; m<nmb; ++m) {
    LogicalLocation &lloc = pm->lloc_eachmb[m + pmbp->gids];
    goffs_h(m,0) = static_cast<int>(lloc.lx1)*(pm->mb_indcs.nx1);
    goffs_h(m,1) = static_cast<int>(lloc.lx2)*(pm->mb_indcs.nx2);
    goffs_h(m,2) = static_cast<int>(lloc.lx3)*(pm->mb_indcs.nx3);
  }
  Kokkos::deep_copy(goffs_, goffs_h);

  // global work arrays and reusable FFT plans (shapes never change)
  Kokkos::realloc(rbuf_a_, nx3_, nx2_, nx1_);
  Kokkos::realloc(rbuf_b_, nx3_, nx2_, nx1_);
  Kokkos::realloc(zin_, nx3_, nx2_, nx1_);
  Kokkos::realloc(zout_, nx3_, nx2_, nx1_);
  Kokkos::realloc(gplane_il_, nx3_, nx2_);
  Kokkos::realloc(gplane_iu_, nx3_, nx2_);
  plan_fwd_ = std::make_unique<FFTPlan>(DevExeSpace(), zin_, zout_,
                KokkosFFT::Direction::forward, KokkosFFT::axis_type<3>({0,1,2}));
  plan_bwd_ = std::make_unique<FFTPlan>(DevExeSpace(), zout_, zin_,
                KokkosFFT::Direction::backward, KokkosFFT::axis_type<3>({0,1,2}));
}

//----------------------------------------------------------------------------------------
//! \fn FFTGravitySolver::ComputeQomt()
//! \brief q*Omega*(time since last shear-periodic instant). The box is strictly
//! shear-periodic at times t_n = n*Ly/(q*Omega*Lx); solving in the frame rolled to the
//! most recent t_n keeps the y-shift (and remap error) minimal.

Real FFTGravitySolver::ComputeQomt(Real time) const {
  if (!shearing_box || qshear == 0.0 || omega0 == 0.0) return 0.0;
  Real tshear = ly_/(qshear*omega0*lx_);
  Real dts = time - std::floor(time/tshear)*tshear;
  return qshear*omega0*dts;
}

//----------------------------------------------------------------------------------------
//! \fn FFTGravitySolver::Solve()
//! \brief main entry: computes Phi from current density, fills pgrav->phi (active zones
//! + one face-adjacent ghost layer). pdriver unused (may be nullptr in pgen calls).

void FFTGravitySolver::Solve(Driver *pdriver, int stage) {
  Real four_pi_G = pmy_pack->pgrav->four_pi_G;
  Real qomt = ComputeQomt(pmy_pack->pmesh->time);

  GatherDensity(four_pi_G);                // pack -> rbuf_a_ = 4piG*rho
  RollUnroll(rbuf_a_, rbuf_b_, qomt);      // roll into periodic frame
  if (vert_bc == FFTVertBC::periodic) {
    SolvePeriodic(qomt);                   // rbuf_b_ -> rbuf_a_ = Phi (rolled)
  } else {
    SolveOpen(qomt);
  }
  RollUnroll(rbuf_a_, rbuf_b_, -qomt);     // unroll back to current frame
  FillShearGhostPlanes(rbuf_b_, qomt);
  ScatterPhi(rbuf_b_);
}

//----------------------------------------------------------------------------------------
//! \fn FFTGravitySolver::GatherDensity()
//! \brief copy active-zone density from all MeshBlocks into the global array

void FFTGravitySolver::GatherDensity(Real four_pi_G) {
  auto u0 = (pmy_pack->pmhd != nullptr) ? pmy_pack->pmhd->u0 : pmy_pack->phydro->u0;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto goffs = goffs_;
  auto rbuf = rbuf_a_;
  par_for("fftg_gather", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    rbuf(goffs(m,2)+(k-ks), goffs(m,1)+(j-js), goffs(m,0)+(i-is)) =
        four_pi_G*u0(m,IDN,k,j,i);
  });
}

//----------------------------------------------------------------------------------------
//! \fn FFTGravitySolver::RollUnroll()
//! \brief conservative per-x-column shift of content by (+sgn_qomt*x1v) in y, with
//! periodic wrap. Rolled frame: y' = y + qomt*x, so rho'(j) = rho(j - qomt*x/dx2)
//! => roll uses sgn_qomt = +qomt, unroll uses -qomt. Integer part of the shift is
//! folded into the (periodic) scratch load; fractional part eps in (-1,1) is applied
//! with the conservative remap-flux kernels shared with orbital advection.

void FFTGravitySolver::RollUnroll(const DvceArray3D<Real> &src, DvceArray3D<Real> &dst,
                                  Real sgn_qomt) {
  if (sgn_qomt == 0.0) {
    Kokkos::deep_copy(dst, src);
    return;
  }
  int nx1 = nx1_, nx2 = nx2_, nx3 = nx3_;
  Real dx1 = dx1_, dx2 = dx2_, x1min = x1min_;
  auto order = remap_order;
  int scr_lvl = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nx2 + 2*PAD)*2;
  par_for_outer("fftg_roll", DevExeSpace(), scr_size, scr_lvl, 0, nx3-1, 0, nx1-1,
  KOKKOS_LAMBDA(TeamMember_t member, const int k, const int i) {
    ScrArray1D<Real> q(member.team_scratch(scr_lvl), nx2 + 2*PAD);
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nx2 + 2*PAD);
    Real x1v = x1min + (static_cast<Real>(i) + 0.5)*dx1;
    Real yshear = sgn_qomt*x1v;
    int joffset = static_cast<int>(yshear/dx2);
    Real eps = fmod(yshear, dx2)/dx2;
    // load y-column with integer shift folded in (periodic wrap)
    par_for_inner(member, 0, nx2 + 2*PAD - 1, [&](const int jf) {
      int jsrc = ((jf - PAD - joffset) % nx2 + nx2) % nx2;
      q(jf) = src(k, jsrc, i);
    });
    member.team_barrier();
    switch (order) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, PAD, PAD+nx2, eps, q, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, PAD, PAD+nx2, eps, q, flx);
        break;
      default:
        PPMX_RemapFlx(member, PAD, PAD+nx2, eps, q, flx);
        break;
    }
    member.team_barrier();
    par_for_inner(member, 0, nx2-1, [&](const int j) {
      dst(k, j, i) = q(j+PAD) - (flx(j+PAD+1) - flx(j+PAD));
    });
  });
}

//----------------------------------------------------------------------------------------
//! \fn FFTGravitySolver::SolvePeriodic()
//! \brief triply-periodic k-space solve of the discrete Poisson equation. Zeroing the
//! k=0 mode is equivalent to subtracting the mean density (Jeans swindle).

void FFTGravitySolver::SolvePeriodic(Real qomt) {
  int nx1 = nx1_, nx2 = nx2_, nx3 = nx3_;
  Real dx1 = dx1_, dx2 = dx2_, dx3 = dx3_;
  Real sfac = qomt*lx_/ly_;   // converts ky mode index jp to the kx' index shift
  auto zin = zin_;
  auto zout = zout_;
  auto rin = rbuf_b_;
  auto rout = rbuf_a_;

  par_for("fftg_r2z", DevExeSpace(), 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    zin(k,j,i) = Kokkos::complex<Real>(rin(k,j,i), 0.0);
  });

  KokkosFFT::execute(*plan_fwd_, zin_, zout_, KokkosFFT::Normalization::backward);

  par_for("fftg_kmul", DevExeSpace(), 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    if (k == 0 && j == 0 && i == 0) {
      zout(0,0,0) = Kokkos::complex<Real>(0.0, 0.0);
      return;
    }
    // signed mode indices; the sign of jp matters in the shear-shifted kx'
    int ip = (i <= nx1/2) ? i : i - nx1;
    int jp = (j <= nx2/2) ? j : j - nx2;
    int kp = (k <= nx3/2) ? k : k - nx3;
    Real kxdx = (static_cast<Real>(ip) + sfac*static_cast<Real>(jp))*2.0*M_PI/nx1;
    Real kydy = static_cast<Real>(jp)*2.0*M_PI/nx2;
    Real kzdz = static_cast<Real>(kp)*2.0*M_PI/nx3;
    // eigenvalue of the 7-point Laplacian (exact inverse of the srcterm stencil)
    Real d = (2.0*cos(kxdx) - 2.0)/(dx1*dx1)
           + (2.0*cos(kydy) - 2.0)/(dx2*dx2)
           + (2.0*cos(kzdz) - 2.0)/(dx3*dx3);
    zout(k,j,i) /= d;
  });

  KokkosFFT::execute(*plan_bwd_, zout_, zin_, KokkosFFT::Normalization::backward);

  par_for("fftg_z2r", DevExeSpace(), 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    rout(k,j,i) = zin(k,j,i).real();
  });
}

//----------------------------------------------------------------------------------------
//! \fn FFTGravitySolver::SolveOpen()
//! \brief open (vacuum) vertical BC via Koyama & Ostriker (2009) two-solve method:
//! solve A (integer kz, periodic image stack) weighted by (1-exp(-kperp*Lz))/2 plus
//! solve B (half-integer kz, alternating image stack) weighted by (1+exp(-kperp*Lz))/2;
//! all vertical images cancel, leaving the isolated-slab potential. The kperp=0 column
//! is carried entirely by solve B (weights 0 and 1), so no mean-density subtraction.

void FFTGravitySolver::SolveOpen(Real qomt) {
  int nx1 = nx1_, nx2 = nx2_, nx3 = nx3_;
  Real dx1 = dx1_, dx2 = dx2_, dx3 = dx3_;
  Real lz = lz_;
  Real sfac = qomt*lx_/ly_;
  auto zin = zin_;
  auto zout = zout_;
  auto rin = rbuf_b_;
  auto rout = rbuf_a_;

  //--- solve A: integer kz
  par_for("fftg_or2z", DevExeSpace(), 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    zin(k,j,i) = Kokkos::complex<Real>(rin(k,j,i), 0.0);
  });

  KokkosFFT::execute(*plan_fwd_, zin_, zout_, KokkosFFT::Normalization::backward);

  par_for("fftg_kmulA", DevExeSpace(), 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    int ip = (i <= nx1/2) ? i : i - nx1;
    int jp = (j <= nx2/2) ? j : j - nx2;
    int kp = (k <= nx3/2) ? k : k - nx3;
    if (ip == 0 && jp == 0) {
      // kperp = 0: A-weight vanishes (and D=0 at the origin); solve B handles this column
      zout(k,j,i) = Kokkos::complex<Real>(0.0, 0.0);
      return;
    }
    Real kxdx = (static_cast<Real>(ip) + sfac*static_cast<Real>(jp))*2.0*M_PI/nx1;
    Real kydy = static_cast<Real>(jp)*2.0*M_PI/nx2;
    Real kzdz = static_cast<Real>(kp)*2.0*M_PI/nx3;
    // continuum kperp in the screening weight (matches Athena-C selfg_fft_disk.c)
    Real kperp = sqrt(SQR(kxdx/dx1) + SQR(kydy/dx2));
    Real d = (2.0*cos(kxdx) - 2.0)/(dx1*dx1)
           + (2.0*cos(kydy) - 2.0)/(dx2*dx2)
           + (2.0*cos(kzdz) - 2.0)/(dx3*dx3);
    zout(k,j,i) *= 0.5*(1.0 - exp(-kperp*lz))/d;
  });

  KokkosFFT::execute(*plan_bwd_, zout_, zin_, KokkosFFT::Normalization::backward);

  par_for("fftg_oz2rA", DevExeSpace(), 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    rout(k,j,i) = zin(k,j,i).real();
  });

  //--- solve B: half-integer kz, via pre-multiplication by exp(-i*pi*zt/Lz) with
  // cell-centered zt = z - x3min = (k+1/2)*dx3
  par_for("fftg_or2zB", DevExeSpace(), 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    Real theta = M_PI*(static_cast<Real>(k) + 0.5)*dx3/lz;
    zin(k,j,i) = Kokkos::complex<Real>(rin(k,j,i)*cos(theta), -rin(k,j,i)*sin(theta));
  });

  KokkosFFT::execute(*plan_fwd_, zin_, zout_, KokkosFFT::Normalization::backward);

  par_for("fftg_kmulB", DevExeSpace(), 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    int ip = (i <= nx1/2) ? i : i - nx1;
    int jp = (j <= nx2/2) ? j : j - nx2;
    Real kxdx = (static_cast<Real>(ip) + sfac*static_cast<Real>(jp))*2.0*M_PI/nx1;
    Real kydy = static_cast<Real>(jp)*2.0*M_PI/nx2;
    // half-integer vertical wavenumber; cos is even and 2pi-periodic so the raw
    // index k is equivalent to a signed one here, and d is strictly negative
    Real kzdz = (static_cast<Real>(k) + 0.5)*2.0*M_PI/nx3;
    Real kperp = sqrt(SQR(kxdx/dx1) + SQR(kydy/dx2));
    Real d = (2.0*cos(kxdx) - 2.0)/(dx1*dx1)
           + (2.0*cos(kydy) - 2.0)/(dx2*dx2)
           + (2.0*cos(kzdz) - 2.0)/(dx3*dx3);
    zout(k,j,i) *= 0.5*(1.0 + exp(-kperp*lz))/d;
  });

  KokkosFFT::execute(*plan_bwd_, zout_, zin_, KokkosFFT::Normalization::backward);

  // recombine: Phi = Phi_A + Re[exp(+i*pi*zt/Lz) * Phi_B]
  par_for("fftg_oz2rB", DevExeSpace(), 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    Real theta = M_PI*(static_cast<Real>(k) + 0.5)*dx3/lz;
    rout(k,j,i) += cos(theta)*zin(k,j,i).real() - sin(theta)*zin(k,j,i).imag();
  });
}

//----------------------------------------------------------------------------------------
//! \fn FFTGravitySolver::FillShearGhostPlanes()
//! \brief x1-face ghost planes of Phi. Shear-periodic identification
//! Phi(x,y) = Phi(x+Lx, y - q*Omega*Lx*t): the inner ghost plane is the outer boundary
//! column with content shifted by +q*Omega*Lx*t (= +qomt*Lx mod Ly), and vice versa.
//! Reduces to a plain periodic copy when qomt = 0.

void FFTGravitySolver::FillShearGhostPlanes(const DvceArray3D<Real> &phi_g, Real qomt) {
  int nx1 = nx1_, nx2 = nx2_, nx3 = nx3_;
  Real dx2 = dx2_;
  Real ysh = qomt*lx_;
  auto order = remap_order;
  auto gil = gplane_il_;
  auto giu = gplane_iu_;
  int scr_lvl = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nx2 + 2*PAD)*2;
  par_for_outer("fftg_shrgh", DevExeSpace(), scr_size, scr_lvl, 0, 1, 0, nx3-1,
  KOKKOS_LAMBDA(TeamMember_t member, const int n, const int k) {
    ScrArray1D<Real> q(member.team_scratch(scr_lvl), nx2 + 2*PAD);
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nx2 + 2*PAD);
    Real yshear = (n == 0) ? ysh : -ysh;
    int isrc = (n == 0) ? nx1-1 : 0;
    int joffset = static_cast<int>(yshear/dx2);
    Real eps = fmod(yshear, dx2)/dx2;
    par_for_inner(member, 0, nx2 + 2*PAD - 1, [&](const int jf) {
      int jsrc = ((jf - PAD - joffset) % nx2 + nx2) % nx2;
      q(jf) = phi_g(k, jsrc, isrc);
    });
    member.team_barrier();
    switch (order) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, PAD, PAD+nx2, eps, q, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, PAD, PAD+nx2, eps, q, flx);
        break;
      default:
        PPMX_RemapFlx(member, PAD, PAD+nx2, eps, q, flx);
        break;
    }
    member.team_barrier();
    par_for_inner(member, 0, nx2-1, [&](const int j) {
      Real val = q(j+PAD) - (flx(j+PAD+1) - flx(j+PAD));
      if (n == 0) {
        gil(k,j) = val;
      } else {
        giu(k,j) = val;
      }
    });
  });
}

//----------------------------------------------------------------------------------------
//! \fn FFTGravitySolver::ScatterPhi()
//! \brief copy global Phi into per-MeshBlock pgrav->phi, filling active zones plus one
//! face-adjacent ghost layer (the contract of the SelfGravity source term). Interior
//! block faces resolve automatically through global indices; physical x1 faces use the
//! shear-periodic planes; x2 wraps; x3 wraps (periodic) or linearly extrapolates (open).

void FFTGravitySolver::ScatterPhi(const DvceArray3D<Real> &phi_g) {
  auto phi = pmy_pack->pgrav->phi;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nx1 = nx1_, nx2 = nx2_, nx3 = nx3_;
  bool open = (vert_bc == FFTVertBC::open);
  auto goffs = goffs_;
  auto gil = gplane_il_;
  auto giu = gplane_iu_;
  par_for("fftg_scatter", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    int gk = goffs(m,2) + (k-ks);
    int gj = goffs(m,1) + (j-js);
    int gi = goffs(m,0) + (i-is);
    int gjw = ((gj % nx2) + nx2) % nx2;                       // x2 always periodic
    int gkc = (gk < 0) ? 0 : ((gk >= nx3) ? nx3-1 : gk);      // clamp for plane reads
    Real val;
    if (gi < 0) {
      val = gil(gkc, gjw);
    } else if (gi >= nx1) {
      val = giu(gkc, gjw);
    } else if (gk < 0) {
      val = open ? 2.0*phi_g(0,gjw,gi) - phi_g(1,gjw,gi) : phi_g(nx3-1,gjw,gi);
    } else if (gk >= nx3) {
      val = open ? 2.0*phi_g(nx3-1,gjw,gi) - phi_g(nx3-2,gjw,gi) : phi_g(0,gjw,gi);
    } else {
      val = phi_g(gk, gjw, gi);
    }
    phi(m,0,k,j,i) = val;
  });
}

} // namespace gravity
