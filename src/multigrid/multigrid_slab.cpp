//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file multigrid_slab.cpp
//! \brief slab-open (vacuum) x3 boundary conditions for the multigrid solver
//! (Phase 3: the stratified shearing box of Shi & Chiang 2014).
//!
//! The domain is (shear-)periodic in x1, periodic in x2, and open (vacuum) in x3.
//! The exact solution of the discrete Poisson equation on an infinite vacuum stack of
//! z-planes separates into horizontal Fourier modes: for each mode the 7-point stencil
//! reduces to a three-term recurrence in z whose decaying solution is mu^{|dn|} with
//!     alpha = 2 - lambda_perp*dz^2,   mu = (alpha - sqrt(alpha^2-4))/2  in (0,1),
//! where lambda_perp <= 0 is the discrete horizontal eigenvalue.  The infinite-domain
//! discrete Green's function is G(dn) = -dz^2 * mu^{|dn|}/sqrt(alpha^2-4), and for the
//! kperp = 0 mode (where alpha = 2 degenerates) G(dn) = +dz^2*|dn|/2, the discrete
//! analogue of the isolated-slab potential 2*pi*G*Sigma*|z| (symmetric gauge).
//!
//! Each Solve() computes the face-value planes
//!     Phi_face(x,y) = sum_n' S(n') * (G(n_last - n') + G(n_ghost - n'))/2 ,
//! (S = 4*pi*G*rho in the strictly periodic "rolled" frame) on both x3 faces and the
//! boundary conditions are imposed multipole-style at every MG level as
//!     ghost = 2*Phi_face - interior ,
//! which makes the converged interior solution *exactly* the restriction of the
//! infinite-domain discrete solution (up to the horizontal remap error of the roll).
//!
//! Shear is handled with the same conventions as fft_gravity.cpp / multigrid_shear.cpp:
//! the density is rolled into the periodic frame with the conservative remap kernels,
//! the shear-shifted radial wavenumber kx' = kx + qomt*(Lx/Ly)*ky enters the horizontal
//! eigenvalue, and the resulting planes are unrolled back.  The plane pyramid spans all
//! block MG levels (pyramid index p = block level shift) and all root MG levels
//! (p = nmblevel-1 + root ll); each level carries ngh-deep ghost rings filled by
//! periodic wrap in x2 and shear-remapped wrap in x1 so that the x3 ghost fill can
//! overwrite the x1/x2 ghost corners consistently (the x3 pass runs last in both the
//! block-level PhysicalBoundary task and MGRootBoundary).
//!
//! Multi-rank: the root-resolution density gather is zero-filled per rank and summed
//! with one MPI_Allreduce; every rank then computes the (identical) planes redundantly
//! with no further communication.  Refined blocks are restricted conservatively into
//! the root sampling during the gather (refinement must stay away from the x3 faces;
//! CheckSlabBlockLevels enforces root level for all x3-boundary blocks).
//!
//! Cost and distribution (2026-09-08 rewrite): no rank ever holds the full 3D density.
//! Each rank transforms only the root k-planes its own MeshBlocks touch (a padded
//! (ny,nx) plane per k-plane, zero outside its blocks), rolls each into the strictly
//! periodic frame as an exact phase exp(-i ky qomt x) between the y and x transforms,
//! and accumulates its share of the two face-plane spectra; ONE Allreduce of those two
//! (ny,nx) complex planes then gives the spectra of the whole box, since the transform
//! and the Green's-weighted sum are linear.  Communication is O(nx ny) per solve instead
//! of O(nx ny nz), and the per-rank work scales with the rank's share of the mesh.  The
//! earlier version gathered and rolled the full density on every rank and transformed
//! every plane redundantly, which made a 67M-cell box cost ~25 s per cycle on 16 nodes.
//! The phase roll replaces the conservative remap of the density planes (limited, hence
//! nonlinear: partial planes did not add up across ranks); the ghost-ring fill of the
//! plane pyramid keeps the remap so the planes match the multigrid's own ghost
//! convention.
//! Requires a build with -D Athena_ENABLE_FFT=ON (kokkos-fft) for the plane
//! computation; the guard lives in MGGravityDriver.

#include <cmath>
#include <iostream>
#include <memory>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../shearing_box/remap_fluxes.hpp"
#include <vector>

#include "multigrid.hpp"

#if FFT_ENABLED
#include <KokkosFFT.hpp>
#endif

namespace {
// halo width for the conservative y-remap scratch arrays (PPMX needs 3; the integer
// part of the shift is folded into the periodic-wrapped scratch load)
constexpr int PAD = 3;
}

//----------------------------------------------------------------------------------------
//! \struct MultigridDriver::MGSlabFFTPlans
//! \brief opaque holder for the reusable kokkos-fft plans (2D transforms of one
//! (ny,nx) slice; created lazily on the first ComputeSlabPlanes call)

struct MultigridDriver::MGSlabFFTPlans {
#if FFT_ENABLED
  using ComplexArray2D = DvceArray2D<Kokkos::complex<Real>>;
  using Plan1D = KokkosFFT::Plan<DevExeSpace, ComplexArray2D, ComplexArray2D, 1>;
  // the (ny,nx) plane is transformed one axis at a time so that the shear roll can be
  // applied as an exact phase between the y and the x transforms
  std::unique_ptr<Plan1D> fwd_y, fwd_x, bwd_y, bwd_x;
#endif
};

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::AllocateSlabPlanes()
//! \brief allocate the plane pyramid and (re)build the per-block offset tables.
//! Idempotent: PrepareForAMR calls it again after every remesh (plane sizes are fixed
//! by the root resolution; only the per-block tables track the current mesh).

void MultigridDriver::AllocateSlabPlanes() {
  if (!mg_slab_enabled_) return;
  int ngh = mgroot_->GetGhostCells();
  auto &mindcs = pmy_mesh_->mesh_indcs;
  int nx = mindcs.nx1, ny = mindcs.nx2, nz = mindcs.nx3;

  if (slab_planes_ == nullptr) {
    int nmbl = mglevels_->GetNumberOfLevels();
    int nrl  = mgroot_->GetNumberOfLevels();
    slab_nplanes_ = nmbl + nrl - 1;
    slab_planes_ = new DvceArray3D<Real>[slab_nplanes_];
    for (int p = 0; p < slab_nplanes_; ++p) {
      Kokkos::realloc(slab_planes_[p], 2, (ny >> p) + 2*ngh, (nx >> p) + 2*ngh);
    }
    Kokkos::realloc(slab_dens_, ny, nx);   // one padded root-resolution plane
    Kokkos::realloc(slab_zin_, ny, nx);
    Kokkos::realloc(slab_zout_, ny, nx);
    Kokkos::realloc(slab_zplanes_, 2, ny, nx);
    Kokkos::realloc(slab_mu_, ny, nx);
    Kokkos::realloc(slab_wt_, ny, nx);
  }

  // per-block tables: root-level (lx1,lx2) for the boundary-condition kernels
  // (x3-boundary blocks are guaranteed to be at root level), and finest-level
  // global cell offsets + refinement level for the density gather
  int nmb = pmy_pack_->nmb_thispack;
  if (static_cast<int>(slab_lloc_.extent(0)) != nmb) {
    Kokkos::realloc(slab_lloc_, nmb, 2);
    Kokkos::realloc(slab_goffs_, nmb, 4);
  }
  auto h_lloc = Kokkos::create_mirror_view(slab_lloc_);
  auto h_goffs = Kokkos::create_mirror_view(slab_goffs_);
  auto &bindcs = pmy_mesh_->mb_indcs;
  for (int m = 0; m < nmb; ++m) {
    LogicalLocation &lloc = pmy_mesh_->lloc_eachmb[m + pmy_pack_->gids];
    int lev = lloc.level - pmy_mesh_->root_level;
    if (lev == 0) {
      h_lloc(m,0) = static_cast<int>(lloc.lx1);
      h_lloc(m,1) = static_cast<int>(lloc.lx2);
    } else {
      h_lloc(m,0) = -1;
      h_lloc(m,1) = -1;
    }
    h_goffs(m,0) = static_cast<int>(lloc.lx1)*bindcs.nx1;
    h_goffs(m,1) = static_cast<int>(lloc.lx2)*bindcs.nx2;
    h_goffs(m,2) = static_cast<int>(lloc.lx3)*bindcs.nx3;
    h_goffs(m,3) = lev;
  }
  Kokkos::deep_copy(slab_lloc_, h_lloc);
  Kokkos::deep_copy(slab_goffs_, h_goffs);
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::CheckSlabBlockLevels()
//! \brief the plane pyramid indexes block levels by their shift relative to the root
//! resolution, which requires every MeshBlock touching an x3 face to be at root level
//! (and consequently no x3-boundary octets exist).  Called at setup and after every
//! AMR remesh.

void MultigridDriver::CheckSlabBlockLevels() {
  if (!mg_slab_enabled_) return;
  for (int m = 0; m < pmy_mesh_->nmb_total; ++m) {
    LogicalLocation &lloc = pmy_mesh_->lloc_eachmb[m];
    std::int32_t nmbx3 =
        (pmy_mesh_->nmb_rootx3 << (lloc.level - pmy_mesh_->root_level));
    if ((lloc.lx3 == 0 || lloc.lx3 == (nmbx3-1)) &&
        lloc.level != pmy_mesh_->root_level) {
      std::cout << "### FATAL ERROR in MultigridDriver::CheckSlabBlockLevels"
                << std::endl
                << "Multigrid slab-open x3 boundaries require all MeshBlocks on the "
                << "x3 faces to remain at the root level. Keep refined regions away "
                << "from the top/bottom of the box (e.g. via refinement criteria or "
                << "static refinement of interior z-rows only)." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::FreeSlabPlanes()
//! \brief release the pyramid and the FFT plans (called from the destructor; lives in
//! this file so MGSlabFFTPlans is a complete type at the delete site)

void MultigridDriver::FreeSlabPlanes() {
  delete [] slab_planes_;
  slab_planes_ = nullptr;
  delete slab_plans_;
  slab_plans_ = nullptr;
}

#if FFT_ENABLED

namespace {

//----------------------------------------------------------------------------------------
//! \fn SlabRemapRow()
//! \brief conservative periodic y-shift of one row: loads src rows [0,ny) of column
//! (view addressed by the caller-supplied lambda), shifts content by +yshear, writes
//! through the dst lambda. Shared by the roll, unroll, and x1 ghost-ring fills.

template <typename SrcLam, typename DstLam>
KOKKOS_INLINE_FUNCTION
void SlabRemapRow(TeamMember_t member, int scr_lvl, int ny, Real dx2, Real yshear,
                  ReconstructionMethod order, const SrcLam &src, const DstLam &dst) {
  ScrArray1D<Real> q(member.team_scratch(scr_lvl), ny + 2*PAD);
  ScrArray1D<Real> flx(member.team_scratch(scr_lvl), ny + 2*PAD);
  int joffset = static_cast<int>(yshear/dx2);
  Real eps = fmod(yshear, dx2)/dx2;
  par_for_inner(member, 0, ny + 2*PAD - 1, [&](const int jf) {
    int jsrc = ((jf - PAD - joffset) % ny + ny) % ny;
    q(jf) = src(jsrc);
  });
  member.team_barrier();
  switch (order) {
    case ReconstructionMethod::dc:
      DC_RemapFlx(member, PAD, PAD+ny, eps, q, flx);
      break;
    case ReconstructionMethod::plm:
      PLM_RemapFlx(member, PAD, PAD+ny, eps, q, flx);
      break;
    default:
      PPMX_RemapFlx(member, PAD, PAD+ny, eps, q, flx);
      break;
  }
  member.team_barrier();
  par_for_inner(member, 0, ny-1, [&](const int j) {
    dst(j, q(j+PAD) - (flx(j+PAD+1) - flx(j+PAD)));
  });
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ComputeSlabPlanes()
//! \brief compute the Dirichlet face-value planes on both x3 faces from the current
//! density and populate the whole plane pyramid (see file header for the method)

void MultigridDriver::ComputeSlabPlanes(const DvceArray5D<Real> &u0, const int ivar,
                                        Real four_pi_G, Real qomt) {
  auto &mindcs = pmy_mesh_->mesh_indcs;
  const int nx = mindcs.nx1, ny = mindcs.nx2, nz = mindcs.nx3;
  auto &msize = pmy_mesh_->mesh_size;
  const Real lx = msize.x1max - msize.x1min;
  const Real ly = msize.x2max - msize.x2min;
  const Real dx1 = lx/static_cast<Real>(nx);
  const Real dx2 = ly/static_cast<Real>(ny);
  const Real dx3 = (msize.x3max - msize.x3min)/static_cast<Real>(nz);
  const Real x1min = msize.x1min;
  const int ngh = mgroot_->GetGhostCells();
  const ReconstructionMethod order = mg_remap_order_;

  // ---- 1-4. per-rank partial face-plane spectra.  Each rank handles only the root
  //         k-planes its own MeshBlocks touch: for each such plane it gathers its cells
  //         into a padded root-resolution plane (zero elsewhere), rolls it into the
  //         strictly periodic frame, transforms it, and accumulates the Green's-weighted
  //         contribution to both face spectra.  The FFT and the accumulation are linear
  //         in the density, so summing these partial spectra over ranks (one Allreduce
  //         of two (ny,nx) complex planes) gives exactly the spectra of the whole box --
  //         without ever assembling the full 3D density on any rank (the previous
  //         gather of (nz,ny,nx) per rank plus its Allreduce and the redundant per-rank
  //         transforms of every plane made the cost grow with the box, not the share).
  if (slab_plans_ == nullptr) {
    slab_plans_ = new MGSlabFFTPlans();
    slab_plans_->fwd_y = std::make_unique<MGSlabFFTPlans::Plan1D>(
        DevExeSpace(), slab_zin_, slab_zout_,
        KokkosFFT::Direction::forward, KokkosFFT::axis_type<1>({0}));
    slab_plans_->fwd_x = std::make_unique<MGSlabFFTPlans::Plan1D>(
        DevExeSpace(), slab_zin_, slab_zout_,
        KokkosFFT::Direction::forward, KokkosFFT::axis_type<1>({1}));
    slab_plans_->bwd_x = std::make_unique<MGSlabFFTPlans::Plan1D>(
        DevExeSpace(), slab_zin_, slab_zout_,
        KokkosFFT::Direction::backward, KokkosFFT::axis_type<1>({1}));
    slab_plans_->bwd_y = std::make_unique<MGSlabFFTPlans::Plan1D>(
        DevExeSpace(), slab_zin_, slab_zout_,
        KokkosFFT::Direction::backward, KokkosFFT::axis_type<1>({0}));
  }

  // per-mode decay factor mu and face weight of the discrete vacuum Green's function
  // (kperp = 0 handled separately in the accumulation)
  {
    auto mu = slab_mu_;
    auto wt = slab_wt_;
    Real sfac = qomt*lx/ly;   // ky mode index -> kx' index shift (rolled frame)
    Real dz2 = dx3*dx3;
    par_for("mgslab_modes", DevExeSpace(), 0, ny-1, 0, nx-1,
    KOKKOS_LAMBDA(const int j, const int i) {
      if (j == 0 && i == 0) {
        mu(0,0) = 1.0;
        wt(0,0) = 0.0;
        return;
      }
      int ip = (i <= nx/2) ? i : i - nx;
      int jp = (j <= ny/2) ? j : j - ny;
      Real kxdx = (static_cast<Real>(ip) + sfac*static_cast<Real>(jp))*2.0*M_PI
                  /static_cast<Real>(nx);
      Real kydy = static_cast<Real>(jp)*2.0*M_PI/static_cast<Real>(ny);
      Real lam = (2.0*cos(kxdx) - 2.0)/(dx1*dx1) + (2.0*cos(kydy) - 2.0)/(dx2*dx2);
      Real alpha = 2.0 - lam*dz2;                    // >= 2, > 2 strictly off-origin
      Real s = sqrt(alpha*alpha - 4.0);
      Real m_ = 2.0/(alpha + s);                     // = (alpha - s)/2, stable form
      mu(j,i) = m_;
      wt(j,i) = -(dz2/s)*0.5*(1.0 + m_);            // face value: mean of the last
    });                                              // center and ghost center
  }

  auto zin = slab_zin_;
  auto zout = slab_zout_;
  auto zplanes = slab_zplanes_;
  Kokkos::deep_copy(zplanes, Kokkos::complex<Real>(0.0, 0.0));
  {
    auto &indcs = pmy_mesh_->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb = pmy_pack_->nmb_thispack;
    int nmb1 = nmb - 1;
    auto goffs = slab_goffs_;
    Real fpg = four_pi_G;
    auto plane = slab_dens_;   // (ny,nx) padded root-resolution plane
    auto mu = slab_mu_;
    auto wt = slab_wt_;
    Real dz2 = dx3*dx3;

    // the distinct root k-planes this rank's blocks cover (host: goffs is small)
    auto h_goffs = Kokkos::create_mirror_view_and_copy(HostMemSpace(), slab_goffs_);
    std::vector<int> kplanes;
    {
      std::vector<char> seen(nz, 0);
      int nmbz = indcs.nx3;
      for (int m = 0; m < nmb; ++m) {
        int lev = h_goffs(m,3);
        int k0 = h_goffs(m,2) >> lev;
        int k1 = (h_goffs(m,2) + nmbz - 1) >> lev;
        for (int kg = k0; kg <= k1; ++kg) seen[kg] = 1;
      }
      for (int kg = 0; kg < nz; ++kg) if (seen[kg]) kplanes.push_back(kg);
    }

    for (int kg : kplanes) {
      // gather this rank's cells of root plane kg (conservative average of refined
      // blocks), zero elsewhere
      Kokkos::deep_copy(plane, 0.0);
      par_for("mgslab_gather", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        int lev = goffs(m,3);
        if (((goffs(m,2) + (k-ks)) >> lev) != kg) return;
        int gi = (goffs(m,0) + (i-is)) >> lev;
        int gj = (goffs(m,1) + (j-js)) >> lev;
        Real w = 1.0/static_cast<Real>(1 << (3*lev));
        Kokkos::atomic_add(&plane(gj,gi), w*fpg*u0(m,ivar,k,j,i));
      });

      // horizontal transform into the strictly periodic (rolled) frame: FFT along y,
      // then the roll rho'(x,y) = rho(x, y - qomt*x) as the exact phase
      // exp(-i ky qomt x) per column, then FFT along x.  Linear in the density (so the
      // per-rank partial planes add up exactly) and free of any remap error.
      par_for("mgslab_r2z", DevExeSpace(), 0, ny-1, 0, nx-1,
      KOKKOS_LAMBDA(const int j, const int i) {
        zin(j,i) = Kokkos::complex<Real>(plane(j,i), 0.0);
      });
      KokkosFFT::execute(*(slab_plans_->fwd_y), slab_zin_, slab_zout_,
                         KokkosFFT::Normalization::backward);
      par_for("mgslab_phase", DevExeSpace(), 0, ny-1, 0, nx-1,
      KOKKOS_LAMBDA(const int j, const int i) {
        int jp = (j <= ny/2) ? j : j - ny;
        Real ky = 2.0*M_PI*static_cast<Real>(jp)/ly;
        Real x1v = x1min + (static_cast<Real>(i) + 0.5)*dx1;
        Real arg = -ky*qomt*x1v;
        zin(j,i) = zout(j,i)*Kokkos::complex<Real>(cos(arg), sin(arg));
      });
      KokkosFFT::execute(*(slab_plans_->fwd_x), slab_zin_, slab_zout_,
                         KokkosFFT::Normalization::backward);
      const int k = kg;
      par_for("mgslab_accum", DevExeSpace(), 0, ny-1, 0, nx-1,
      KOKKOS_LAMBDA(const int j, const int i) {
        if (j == 0 && i == 0) {
          // kperp = 0: discrete isolated-slab potential, symmetric gauge
          zplanes(0,0,0) += zout(0,0)*(0.5*dz2*(static_cast<Real>(k) + 0.5));
          zplanes(1,0,0) += zout(0,0)*(0.5*dz2*(static_cast<Real>(nz-k) - 0.5));
        } else {
          zplanes(0,j,i) += zout(j,i)*(wt(j,i)*pow(mu(j,i), k));
          zplanes(1,j,i) += zout(j,i)*(wt(j,i)*pow(mu(j,i), nz-1-k));
        }
      });
    }
  }
#if MPI_PARALLEL_ENABLED
  // sum the partial face spectra over ranks (two (ny,nx) complex planes)
  Kokkos::fence();
  MPI_Allreduce(MPI_IN_PLACE, reinterpret_cast<Real*>(zplanes.data()),
                static_cast<int>(2*zplanes.size()), MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  // ---- 5-6. inverse transform of the two face-plane spectra: iFFT along x, the
  //         unroll as the conjugate phase, iFFT along y -> real planes in the current
  //         frame, into the interior of pyramid level 0
  {
    auto plane0 = slab_planes_[0];
    for (int f = 0; f < 2; ++f) {
      par_for("mgslab_pcopy", DevExeSpace(), 0, ny-1, 0, nx-1,
      KOKKOS_LAMBDA(const int j, const int i) {
        zin(j,i) = zplanes(f,j,i);
      });
      KokkosFFT::execute(*(slab_plans_->bwd_x), slab_zin_, slab_zout_,
                         KokkosFFT::Normalization::backward);
      par_for("mgslab_unphase", DevExeSpace(), 0, ny-1, 0, nx-1,
      KOKKOS_LAMBDA(const int j, const int i) {
        int jp = (j <= ny/2) ? j : j - ny;
        Real ky = 2.0*M_PI*static_cast<Real>(jp)/ly;
        Real x1v = x1min + (static_cast<Real>(i) + 0.5)*dx1;
        Real arg = ky*qomt*x1v;
        zin(j,i) = zout(j,i)*Kokkos::complex<Real>(cos(arg), sin(arg));
      });
      KokkosFFT::execute(*(slab_plans_->bwd_y), slab_zin_, slab_zout_,
                         KokkosFFT::Normalization::backward);
      par_for("mgslab_z2r", DevExeSpace(), 0, ny-1, 0, nx-1,
      KOKKOS_LAMBDA(const int j, const int i) {
        plane0(f, ngh+j, ngh+i) = zout(j,i).real();
      });
    }
  }

  // ---- 7. ghost rings + restriction down the pyramid.  Ring fill: x1 ghost columns
  //         by shear-remapped wrap (plain wrap when qomt = 0), then x2 rows by
  //         periodic wrap (covering the corners).
  for (int p = 0; p < slab_nplanes_; ++p) {
    int nxp = nx >> p, nyp = ny >> p;
    Real dx2p = ly/static_cast<Real>(nyp);
    auto plane = slab_planes_[p];
    {
      int scr_lvl = 0;
      size_t scr_size = ScrArray1D<Real>::shmem_size(nyp + 2*PAD)*2;
      Real ysh = qomt*lx;
      par_for_outer("mgslab_ringx1", DevExeSpace(), scr_size, scr_lvl,
                    0, 1, 0, 2*ngh-1,
      KOKKOS_LAMBDA(TeamMember_t member, const int f, const int c) {
        int side = c/ngh, n = c - side*ngh;
        int isrc = (side == 0) ? (ngh + nxp - 1 - n) : (ngh + n);
        int idst = (side == 0) ? (ngh - 1 - n) : (ngh + nxp + n);
        Real yshear = (side == 0) ? ysh : -ysh;
        SlabRemapRow(member, scr_lvl, nyp, dx2p, yshear, order,
                     [&](int jsrc) { return plane(f, ngh+jsrc, isrc); },
                     [&](int j, Real v) { plane(f, ngh+j, idst) = v; });
      });
    }
    par_for("mgslab_ringx2", DevExeSpace(), 0, 1, 0, ngh-1, 0, nxp+2*ngh-1,
    KOKKOS_LAMBDA(const int f, const int n, const int i) {
      plane(f, ngh-1-n, i) = plane(f, ngh+nyp-1-n, i);
      plane(f, ngh+nyp+n, i) = plane(f, ngh+n, i);
    });
    if (p < slab_nplanes_-1) {
      auto planec = slab_planes_[p+1];
      par_for("mgslab_restrict", DevExeSpace(), 0, 1, 0, (nyp>>1)-1, 0, (nxp>>1)-1,
      KOKKOS_LAMBDA(const int f, const int jc, const int ic) {
        planec(f, ngh+jc, ngh+ic) = 0.25*(plane(f, ngh+2*jc,   ngh+2*ic)
                                        + plane(f, ngh+2*jc,   ngh+2*ic+1)
                                        + plane(f, ngh+2*jc+1, ngh+2*ic)
                                        + plane(f, ngh+2*jc+1, ngh+2*ic+1));
      });
    }
  }
}

#else  // !FFT_ENABLED

void MultigridDriver::ComputeSlabPlanes(const DvceArray5D<Real> &u0, const int ivar,
                                        Real four_pi_G, Real qomt) {
  std::cout << "### FATAL ERROR in MultigridDriver::ComputeSlabPlanes" << std::endl
            << "Slab-open x3 boundaries require a build with -D Athena_ENABLE_FFT=ON"
            << std::endl;
  std::exit(EXIT_FAILURE);
}

#endif  // FFT_ENABLED
