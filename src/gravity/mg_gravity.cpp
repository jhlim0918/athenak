//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.cpp
//! \brief create multigrid solver for gravity

// C headers

// C++ headers
#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mhd/mhd.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "gravity.hpp"
#include "mg_gravity.hpp"
#include "../driver/driver.hpp"

class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::MGGravityDriver(Mesh *pm, ParameterInput *pin)
//! \brief MGGravityDriver constructor

MGGravityDriver::MGGravityDriver(MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
    four_pi_G_ = pin->GetOrAddReal("gravity", "four_pi_G", -1.0);
    omega_ = pin->GetOrAddReal("gravity", "omega", 1.15);
    eps_ = pin->GetOrAddReal("gravity", "threshold", -1.0);
    niter_ = pin->GetOrAddInteger("gravity", "niteration", -1);
    npresmooth_ = pin->GetOrAddReal("gravity", "npresmooth", npresmooth_);
    npostsmooth_ = pin->GetOrAddReal("gravity", "npostsmooth", npostsmooth_);
    full_multigrid_ = pin->GetOrAddBoolean("gravity", "full_multigrid", false);
    fmg_ncycle_ = pin->GetOrAddInteger("gravity", "fmg_ncycle", 1);
    fshowdef_ = pin->GetOrAddInteger("gravity", "show_defect", 0);
    mg_verbose_ = pin->GetOrAddInteger("gravity", "mg_verbose", 0);
    fsubtract_average_ = pin->GetOrAddBoolean("gravity", "subtract_average", true);
    if (eps_ < 0.0 && niter_ < 0) {
        std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <gravity> block." << std::endl
        << "When both parameters are specified, \"niteration\" is ignored." << std::endl
        << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
        exit(EXIT_FAILURE);
  }
  if (four_pi_G_ < 0.0) {
    std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
    exit(EXIT_FAILURE);
  }
  // Override MG boundary conditions if specified in input file.
  // Options: "zerofixed" (Dirichlet zero), "zerograd" (Neumann zero), "multipole",
  // "slab" (open/vacuum x3 of a horizontally (shear-)periodic box, Phase 3).
  std::string mg_bc_str = pin->GetOrAddString("gravity", "mg_bc", "none");
  if (mg_bc_str != "none") {
    BoundaryFlag mg_bc;
    if (mg_bc_str == "zerofixed") {
      mg_bc = BoundaryFlag::mg_zerofixed;
    } else if (mg_bc_str == "zerograd") {
      mg_bc = BoundaryFlag::mg_zerograd;
    } else if (mg_bc_str == "multipole") {
      mg_bc = BoundaryFlag::mg_multipole;
    } else if (mg_bc_str == "slab") {
      mg_bc = BoundaryFlag::mg_slab;
    } else {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "Unknown mg_bc type: " << mg_bc_str << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int f = 0; f < 6; ++f) {
      if (mg_mesh_bcs_[f] != BoundaryFlag::periodic &&
          mg_mesh_bcs_[f] != BoundaryFlag::shear_periodic) {
        mg_mesh_bcs_[f] = mg_bc;
      }
    }
  }

  // Slab-open (vacuum) x3 boundaries: valid only on both x3 faces of a 3D box that
  // is (shear-)periodic in x1 and periodic in x2 (see multigrid_slab.cpp).
  mg_slab_enabled_ = (mg_mesh_bcs_[BoundaryFace::inner_x3] == BoundaryFlag::mg_slab ||
                      mg_mesh_bcs_[BoundaryFace::outer_x3] == BoundaryFlag::mg_slab);
  if (mg_slab_enabled_) {
    auto ok_x1 = [](BoundaryFlag f) {
      return (f == BoundaryFlag::periodic || f == BoundaryFlag::shear_periodic);
    };
    if (mg_mesh_bcs_[BoundaryFace::inner_x3] != BoundaryFlag::mg_slab ||
        mg_mesh_bcs_[BoundaryFace::outer_x3] != BoundaryFlag::mg_slab ||
        !ok_x1(mg_mesh_bcs_[BoundaryFace::inner_x1]) ||
        !ok_x1(mg_mesh_bcs_[BoundaryFace::outer_x1]) ||
        mg_mesh_bcs_[BoundaryFace::inner_x2] != BoundaryFlag::periodic ||
        mg_mesh_bcs_[BoundaryFace::outer_x2] != BoundaryFlag::periodic) {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "<gravity> mg_bc = slab requires non-periodic x3 on BOTH faces, "
                << "(shear-)periodic x1, and periodic x2 boundaries." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!pmy_mesh_->three_d) {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "<gravity> mg_bc = slab requires a 3D mesh." << std::endl;
      std::exit(EXIT_FAILURE);
    }
#if !FFT_ENABLED
    std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
              << "<gravity> mg_bc = slab requires a build with "
              << "-D Athena_ENABLE_FFT=ON (kokkos-fft)." << std::endl;
    std::exit(EXIT_FAILURE);
#endif
  }
  // Shear-periodic x1 boundaries (Phase 1: uniform grid, single rank).
  // mesh.cpp guarantees that shear_periodic appears on both x1 faces or neither,
  // and only together with a <shearing_box> input block.
  mg_shear_enabled_ =
      (pmy_mesh_->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic);
  if (mg_shear_enabled_) {
    mg_qshear_ = pin->GetReal("shearing_box", "qshear");
    mg_omega0_ = pin->GetReal("shearing_box", "omega0");
    std::string rmap = pin->GetOrAddString("gravity", "mg_remap", "plm");
    if (rmap == "dc") {
      mg_remap_order_ = ReconstructionMethod::dc;
    } else if (rmap == "plm") {
      mg_remap_order_ = ReconstructionMethod::plm;
    } else if (rmap == "ppmx") {
      mg_remap_order_ = ReconstructionMethod::ppmx;
    } else {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "<gravity> mg_remap = '" << rmap << "' not recognized "
                << "(must be 'dc', 'plm', or 'ppmx')" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Phase 2a/2b: static refinement in the shearing box. The MeshBlocks on the
    // shear-periodic x1 faces must share ONE uniform level (the same invariant the
    // hydro side enforces in Mesh::CheckShearingBoxRefinement): the block shear
    // planes are sized at that level, and the octet shear fill (Phase 2b) relies on
    // the boundary octets tiling the full y-z extent at every octet level. Interior
    // rings (boundary at root, Phase 2a) and uniformly refined boundary annuli
    // (boundary above root, Phase 2b) both pass.
    if (pmy_mesh_->multilevel) {
      // Adaptive refinement is supported (Phase 2c): PrepareForAMR rebuilds the
      // shear planes, per-block offsets, and octet face flags after every remesh,
      // and Mesh::CheckShearingBoxRefinement re-enforces the invariant below after
      // every AMR update.
      int blev_min = std::numeric_limits<int>::max();
      int blev_max = std::numeric_limits<int>::min();
      for (int m = 0; m < pmy_mesh_->nmb_total; ++m) {
        LogicalLocation &lloc = pmy_mesh_->lloc_eachmb[m];
        std::int32_t nmbx1 =
            (pmy_mesh_->nmb_rootx1 << (lloc.level - pmy_mesh_->root_level));
        if (lloc.lx1 == 0 || lloc.lx1 == (nmbx1-1)) {
          blev_min = std::min(blev_min, static_cast<int>(lloc.level));
          blev_max = std::max(blev_max, static_cast<int>(lloc.level));
        }
      }
      if (blev_min != blev_max) {
        std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                  << "Multigrid gravity with shear-periodic boundaries requires all "
                  << "MeshBlocks on the x1 boundaries to share ONE uniform level "
                  << "(levels " << blev_min << ".." << blev_max << " found). Keep "
                  << "refined regions interior in x1, or refine BOTH x1 faces "
                  << "uniformly (full x2/x3 extent, same level)." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    if (pmy_mesh_->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic ||
        pmy_mesh_->mesh_bcs[BoundaryFace::outer_x2] != BoundaryFlag::periodic ||
        (!mg_slab_enabled_ &&
         (pmy_mesh_->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic ||
          pmy_mesh_->mesh_bcs[BoundaryFace::outer_x3] != BoundaryFlag::periodic))) {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "Multigrid gravity with shear-periodic x1 requires periodic "
                << "x2 and periodic (or mg_bc = slab) x3 boundaries." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (mg_bc_str != "none" && mg_bc_str != "slab") {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "<gravity> mg_bc = '" << mg_bc_str << "' cannot be combined with "
                << "shear-periodic x1 boundaries (only 'slab' is supported)."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Mean-source subtraction is required for solvability whenever the domain is fully
  // (shear-)periodic; disable it only when some face is a genuine physical boundary
  // (shear_periodic is a periodic identification, so it does not count as one).
  {
    bool wrap_all = true;
    for (int f = 0; f < 6; ++f) {
      BoundaryFlag mbc = pmy_mesh_->mesh_bcs[f];
      wrap_all = wrap_all && (mbc == BoundaryFlag::periodic ||
                              mbc == BoundaryFlag::shear_periodic);
    }
    if (!wrap_all) {
      fsubtract_average_ = false;
    }
  }

  // Check if multipole BCs are active and configure
  for (int f = 0; f < 6; ++f) {
    if (mg_mesh_bcs_[f] == BoundaryFlag::mg_multipole) {
      mporder_ = 0;  // mark as detected
      break;
    }
  }
  if (mporder_ >= 0) {
    mporder_ = pin->GetOrAddInteger("gravity", "mporder", 4);
    autompo_ = pin->GetOrAddBoolean("gravity", "auto_mporigin", true);
    nodipole_ = pin->GetOrAddBoolean("gravity", "nodipole", false);
    if (mporder_ != 2 && mporder_ != 4) {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "mporder must be 2 (quadrupole) or 4 (hexadecapole)." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (autompo_ && nodipole_) {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "auto_mporigin and nodipole cannot be used together." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!autompo_) {
      mpo_[0] = pin->GetReal("gravity", "mporigin_x1");
      mpo_[1] = pin->GetReal("gravity", "mporigin_x2");
      mpo_[2] = pin->GetReal("gravity", "mporigin_x3");
    }
    AllocateMultipoleCoefficients();
    fsubtract_average_ = false;
  }

  // Source masking parameters
  mask_radius_ = pin->GetOrAddReal("gravity", "mask_radius", -1.0);
  mask_origin_[0] = pin->GetOrAddReal("gravity", "mask_origin_x1", 0.0);
  mask_origin_[1] = pin->GetOrAddReal("gravity", "mask_origin_x2", 0.0);
  mask_origin_[2] = pin->GetOrAddReal("gravity", "mask_origin_x3", 0.0);

  // Allocate the root multigrid
  int nghost = pin->GetOrAddInteger("gravity", "mg_nghost", 1);
  bool root_on_host = pin->GetOrAddBoolean("gravity", "root_on_host", false);
  if ((mg_shear_enabled_ || mg_slab_enabled_) && root_on_host) {
    std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
              << "<gravity> root_on_host is not supported with shear-periodic or "
              << "slab boundaries (their root fills run on device)." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((mg_shear_enabled_ || mg_slab_enabled_) && nghost > 1) {
    std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
              << "<gravity> mg_nghost > 1 is untested with shear-periodic or slab "
              << "boundaries." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  mgroot_ = new MGGravity(this, nullptr, nghost, root_on_host);
  mglevels_ = new MGGravity(this, pmbp, nghost);
  if (mg_shear_enabled_) {
    mglevels_->AllocateShearPlanes();
  }
  // allocate boundary buffers
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers((nvar_));
  mglevels_->pbval->RemapIndicesForMG();
  mglevels_->pbval->ComputePerLevelIndices();
  if (mg_slab_enabled_) {
    CheckSlabBlockLevels();
    AllocateSlabPlanes();
  }
}


//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::~MGGravityDriver()
//! \brief MGGravityDriver destructor

MGGravityDriver::~MGGravityDriver() {
  delete mgroot_;
  delete mglevels_;
}

void MGGravityDriver::SetFourPiG(Real four_pi_G) {
  four_pi_G_ = four_pi_G;
}

//----------------------------------------------------------------------------------------
//! \fn MGGravity::MGGravity(MultigridDriver *pmd, MeshBlock *pmb)
//! \brief MGGravity constructor

MGGravity::MGGravity(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}


//----------------------------------------------------------------------------------------
//! \fn MGGravity::~MGGravity()
//! \brief MGGravity deconstructor

MGGravity::~MGGravity() {
  //delete pmgbval;
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::Solve(int stage, Real dt)
//! \brief load the data and solve

void MGGravityDriver::Solve(Driver *pdriver, int stage, Real dt) {
  RegionIndcs &indcs_ = pmy_pack_->pmesh->mb_indcs;

  // Freeze the shear phase once per solve. Uses pmesh->time (not time+dt), the same
  // convention as the FFT solver, so both solvers see identical boundary
  // identifications at any instant. Note hydro's shear-periodic BC uses time+dt on
  // the final RK stage (hydro_tasks.cpp); the O(dt) offset is a pre-existing,
  // FFT-validated convention -- do not "fix" it here.
  mg_qomt_ = ComputeShearQomt(pmy_pack_->pmesh->time);

  // Reallocate MG arrays and phi if AMR has changed the mesh
  PrepareForAMR();
  {
    int nmb = pmy_pack_->nmb_thispack;
    if (static_cast<int>(pmy_pack_->pgrav->phi.extent_int(0)) != nmb) {
      int ncells1 = indcs_.nx1 + 2*indcs_.ng;
      int ncells2 = (indcs_.nx2 > 1) ? (indcs_.nx2 + 2*indcs_.ng) : 1;
      int ncells3 = (indcs_.nx3 > 1) ? (indcs_.nx3 + 2*indcs_.ng) : 1;
      Kokkos::realloc(pmy_pack_->pgrav->phi, nmb, 1, ncells3, ncells2, ncells1);
    }
  }

  // mglevels_ points to the Multigrid object for all MeshBlocks
  // The MG smoother solves -∇²u = src (note the minus sign from the Laplacian
  // convention: Laplacian(u) = 6u - neighbors = -dx²∇²u).  To obtain the
  // standard Poisson equation ∇²φ = 4πGρ we must load the source with a
  // negative sign so that -∇²φ = -4πGρ, i.e. ∇²φ = +4πGρ.
  // the source density: the gas conserved array, or gas + registered extra density
  // (Gravity::SourceArray, dust track)
  auto &u0 = pmy_pack_->pgrav->SourceArray();
  const int isrc = pmy_pack_->pgrav->SourceIndex();

  // Slab-open x3: recompute the Dirichlet face planes from the current density,
  // rolled by the same frozen shear phase as the x1 ghost fills (mg_qomt_)
  if (mg_slab_enabled_) {
    ComputeSlabPlanes(u0, isrc, four_pi_G_, mg_qomt_);
  }

  mglevels_->LoadSource(u0, isrc, indcs_.ng, -four_pi_G_);

  // Apply source mask (zero source outside mask_radius_)
  mglevels_->ApplyMask();

  // iterative mode - load initial guess
  if (!full_multigrid_) {
    mglevels_->LoadFinestData(pmy_pack_->pgrav->phi, 0, indcs_.ng);
  }

  // Finalize setup (SubtractAverage, level counts) after data is loaded
  SetupMultigrid(dt, false);

  // Compute multipole coefficients for isolated boundaries
  if (mporder_ > 0) {
    if (autompo_) CalculateCenterOfMass();
    CalculateMultipoleCoefficients();
    SyncMultipoleToDevice();
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  if (full_multigrid_)
    SolveFMG(pdriver);
  else
    SolveMG(pdriver);

  Kokkos::fence();

  if (fshowdef_ >= 1) {
    auto t_end = std::chrono::high_resolution_clock::now();
    double mg_elapsed = std::chrono::duration<double>(t_end - t_start).count();
    Real def = 0.0;
    for (int v = 0; v < nvar_; ++v) {
      def += CalculateDefectNorm(MGNormType::l2, v);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "mg_solve_time = " << std::scientific << std::setprecision(6)
                << mg_elapsed << std::endl;
      std::cout << "MGGravityDriver::Solve: Final defect norm = " << def << std::endl;
    }
  }

  mglevels_->RetrieveResult(pmy_pack_->pgrav->phi, 0, indcs_.ng);

  return;
}

void MGGravity::SmoothPack(int color) {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{static_cast<MGGravityDriver*>(pmy_driver_)->omega_/6.0};
  if (on_host_) {
    Smooth(u_[current_level_].h_view, src_[current_level_].h_view,
           coeff_[current_level_].h_view, matrix_[current_level_].h_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  } else {
    Smooth(u_[current_level_].d_view, src_[current_level_].d_view,
           coeff_[current_level_].d_view, matrix_[current_level_].d_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  }
}

void MGGravity::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{0.0};
  if (on_host_) {
    CalculateDefect(def_[current_level_].h_view, u_[current_level_].h_view,
                    src_[current_level_].h_view, coeff_[current_level_].h_view,
                    matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateDefect(def_[current_level_].d_view, u_[current_level_].d_view,
                    src_[current_level_].d_view, coeff_[current_level_].d_view,
                    matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}

void MGGravity::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{0.0};
  if (on_host_) {
    CalculateFASRHS(src_[current_level_].h_view, u_[current_level_].h_view,
                    coeff_[current_level_].h_view, matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateFASRHS(src_[current_level_].d_view, u_[current_level_].d_view,
                    coeff_[current_level_].d_view, matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}


//----------------------------------------------------------------------------------------
// Host-side octet physics for MGGravityDriver

static inline Real OctLaplacian(const MGOctet &o, int v, int k, int j, int i) {
  return (6.0*o.U(v,k,j,i) - o.U(v,k+1,j,i) - o.U(v,k,j+1,i)
          - o.U(v,k,j,i+1) - o.U(v,k-1,j,i) - o.U(v,k,j-1,i)
          - o.U(v,k,j,i-1));
}

void MGGravityDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real dx2 = dx * dx;
  Real isix = omega_ / 6.0;
  int c = color ^ coffset_;
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh + ((c^k^j)&1); i <= ngh+1; i += 2) {
        Real lap = OctLaplacian(oct, 0, k, j, i);
        oct.U(0,k,j,i) -= (lap - oct.Src(0,k,j,i)*dx2)*isix;
      }
    }
  }
}

void MGGravityDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Def(0,k,j,i) = oct.Src(0,k,j,i) - OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}

void MGGravityDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Src(0,k,j,i) += OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::ProlongateOctetBoundariesFluxCons(...)
//! \brief Conservative prolongation of face ghost cells at fine-coarse level boundaries.
//!        Implements the "Conservative Formulation" from Tomida & Stone (2023) Eq. 24-27.
//!        Ghost = (2/3)*coarse_interpolated + (1/3)*fine_active, where transverse
//!        gradients from the coarse buffer provide sub-cell interpolation.
//!        Only face neighbors are handled
//!        (edges/corners are unused by the 7-point stencil).

void MGGravityDriver::ProlongateOctetBoundariesFluxCons(
    MGOctet &oct, std::vector<Real> &cbuf,
    const std::vector<bool> &ncoarse) {
  constexpr Real ot = 1.0/3.0;
  const int ngh = mgroot_->GetGhostCells();
  const int l = ngh, r = ngh + 1;

  // x1 face
  for (int ox1 = -1; ox1 <= 1; ox1 += 2) {
    if (ncoarse[1*9 + 1*3 + (ox1+1)]) {
      int i, fi, fig;
      if (ox1 > 0) {
        i = ngh + 1; fi = ngh + 1; fig = ngh + 2;
      } else {
        i = ngh - 1; fi = ngh;     fig = ngh - 1;
      }
      Real ccval = BufRef(cbuf, 3, 0, ngh, ngh, i);
      Real gx2c = 0.125*(BufRef(cbuf, 3, 0, ngh, ngh+1, i)
                        - BufRef(cbuf, 3, 0, ngh, ngh-1, i));
      Real gx3c = 0.125*(BufRef(cbuf, 3, 0, ngh+1, ngh, i)
                        - BufRef(cbuf, 3, 0, ngh-1, ngh, i));
      oct.U(0, l, l, fig) = ot*(2.0*(ccval - gx2c - gx3c) + oct.U(0, l, l, fi));
      oct.U(0, l, r, fig) = ot*(2.0*(ccval + gx2c - gx3c) + oct.U(0, l, r, fi));
      oct.U(0, r, l, fig) = ot*(2.0*(ccval - gx2c + gx3c) + oct.U(0, r, l, fi));
      oct.U(0, r, r, fig) = ot*(2.0*(ccval + gx2c + gx3c) + oct.U(0, r, r, fi));
    }
  }

  // x2 face
  for (int ox2 = -1; ox2 <= 1; ox2 += 2) {
    if (ncoarse[1*9 + (ox2+1)*3 + 1]) {
      int j, fj, fjg;
      if (ox2 > 0) {
        j = ngh + 1; fj = ngh + 1; fjg = ngh + 2;
      } else {
        j = ngh - 1; fj = ngh;     fjg = ngh - 1;
      }
      Real ccval = BufRef(cbuf, 3, 0, ngh, j, ngh);
      Real gx1c = 0.125*(BufRef(cbuf, 3, 0, ngh, j, ngh+1)
                        - BufRef(cbuf, 3, 0, ngh, j, ngh-1));
      Real gx3c = 0.125*(BufRef(cbuf, 3, 0, ngh+1, j, ngh)
                        - BufRef(cbuf, 3, 0, ngh-1, j, ngh));
      oct.U(0, l, fjg, l) = ot*(2.0*(ccval - gx1c - gx3c) + oct.U(0, l, fj, l));
      oct.U(0, l, fjg, r) = ot*(2.0*(ccval + gx1c - gx3c) + oct.U(0, l, fj, r));
      oct.U(0, r, fjg, l) = ot*(2.0*(ccval - gx1c + gx3c) + oct.U(0, r, fj, l));
      oct.U(0, r, fjg, r) = ot*(2.0*(ccval + gx1c + gx3c) + oct.U(0, r, fj, r));
    }
  }

  // x3 face
  for (int ox3 = -1; ox3 <= 1; ox3 += 2) {
    if (ncoarse[(ox3+1)*9 + 1*3 + 1]) {
      int k, fk, fkg;
      if (ox3 > 0) {
        k = ngh + 1; fk = ngh + 1; fkg = ngh + 2;
      } else {
        k = ngh - 1; fk = ngh;     fkg = ngh - 1;
      }
      Real ccval = BufRef(cbuf, 3, 0, k, ngh, ngh);
      Real gx1c = 0.125*(BufRef(cbuf, 3, 0, k, ngh, ngh+1)
                        - BufRef(cbuf, 3, 0, k, ngh, ngh-1));
      Real gx2c = 0.125*(BufRef(cbuf, 3, 0, k, ngh+1, ngh)
                        - BufRef(cbuf, 3, 0, k, ngh-1, ngh));
      oct.U(0, fkg, l, l) = ot*(2.0*(ccval - gx1c - gx2c) + oct.U(0, fk, l, l));
      oct.U(0, fkg, l, r) = ot*(2.0*(ccval + gx1c - gx2c) + oct.U(0, fk, l, r));
      oct.U(0, fkg, r, l) = ot*(2.0*(ccval - gx1c + gx2c) + oct.U(0, fk, r, l));
      oct.U(0, fkg, r, r) = ot*(2.0*(ccval + gx1c + gx2c) + oct.U(0, fk, r, r));
    }
  }
}
