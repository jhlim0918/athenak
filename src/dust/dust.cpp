//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust.cpp
//! \brief implementation of DustGasDrag class constructor and support functions

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "gravity/gravity.hpp"
#include "shearing_box/shearing_box.hpp"
#include "dust.hpp"

namespace dust {
//----------------------------------------------------------------------------------------
// constructor: parses input, validates configuration, allocates deposit fields and
// boundary communication objects

DustGasDrag::DustGasDrag(MeshBlockPack *ppack, ParameterInput *pin) :
    taus("dust_taus",1),
    qdep("qdep",1,1,1,1,1),
    ustar("ustar",1,1,1,1,1),
    dmom("dmom",1,1,1,1,1),
    cdummy("cdum",1,1,1,1,1),
    solver_r("dust_solver_r",1,1,1,1,1),
    solver_p("dust_solver_p",1,1,1,1,1),
    solver_ap("dust_solver_ap",1,1,1,1,1),
    pmy_pack(ppack) {
  // (1) validate configuration ----------------------------------------------------------
  hydro::Hydro *phyd = pmy_pack->phydro;
  particles::Particles *ppar = pmy_pack->ppart;
  if (phyd == nullptr || ppar == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust> block requires both <hydro> and <particles> blocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust drag currently only couples to Hydro, but <mhd> block detected"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ppar->particle_type != ParticleType::dust) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust> block requires <particles>/particle_type = dust" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Driver is constructed after physics modules, so read the coupling and integrator
  // directly from the input while validating the module configuration.
  bool pc2_only = false;
  {
    std::string method = pin->GetOrAddString("dust", "coupling", "imex");
    if (method.compare("imex") == 0) {
      coupling = DustCoupling::imex;
    } else if (method.compare("hybrid") == 0) {
      coupling = DustCoupling::hybrid;
    } else if (method.compare("pc2") == 0) {
      // PC2 uses the hybrid task path, but this direct spelling disables the
      // automatic split-BE fallback and selects PC2 for every cycle.
      coupling = DustCoupling::hybrid;
      pc2_only = true;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/coupling = '" << method
                << "' not recognized (must be imex, pc2, or hybrid)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  std::string integrator = pin->GetOrAddString("time", "integrator", "rk2");
  bool bad_integrator = ((coupling == DustCoupling::imex &&
                          integrator.compare("imex2+") != 0) ||
                         (coupling == DustCoupling::hybrid &&
                          integrator.compare("rk2") != 0));
  if (bad_integrator) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust coupling=imex requires integrator=imex2+, while coupling=pc2 "
              << "or hybrid requires integrator=rk2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmesh->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust drag does not support SMR/AMR" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool shear_x1 = (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] ==
                   BoundaryFlag::shear_periodic);
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    // exceptions: 3D shearing box with shear-periodic x1 (and periodic x2), and, in 3D,
    // physical (non-periodic) x3 faces such as the outflow faces of a self-gravitating
    // slab (Phase 4c).  Ghost deposits landing beyond a physical face have no receiver
    // and are discarded; a particle that leaves the mesh through one is not routed
    // (the deposit halo check aborts on it) -- the vertical policy is a later phase.
    auto &bcs = pmy_pack->pmesh->mesh_bcs;
    bool x1_ok = (bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic) || shear_x1;
    bool x2_ok = (bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic) &&
                 (bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic);
    bool x3_periodic = (bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) &&
                       (bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic);
    if (!(pmy_pack->pmesh->three_d && x1_ok && x2_ok)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dust drag requires periodic boundaries in all "
                << "directions, or in 3D: periodic/shear-periodic x1, periodic x2 and "
                << "any x3" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    physical_faces = !x3_periodic;
    if (!x3_periodic && global_variable::my_rank == 0) {
      std::cout << "# dust: physical x3 boundaries: ghost deposits beyond the x3 faces "
                << "are discarded and a particle crossing an x3 face is removed "
                << "(outflow, Athena-C zbc_out = 1); removed mass is accounted in "
                << "DUST_ESCAPE_SUMMARY" << std::endl;
    }
  }
  // y-remap order of the shear-periodic fold of ghost deposits across the radial faces
  // (MeshBoundaryValuesDep::FoldShearDeposit); inert without shear-periodic x1
  {
    std::string rmap = pin->GetOrAddString("dust","shear_remap","plm");
    if (rmap == "dc") {
      shear_remap = ReconstructionMethod::dc;
    } else if (rmap == "plm") {
      shear_remap = ReconstructionMethod::plm;
    } else if (rmap == "ppmx") {
      shear_remap = ReconstructionMethod::ppmx;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/shear_remap = '" << rmap
                << "' not recognized (must be dc, plm, or ppmx)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    shear_fold = pin->GetOrAddBoolean("dust","shear_fold",true);
    if (shear_x1 && pmy_pack->pmesh->three_d && global_variable::my_rank == 0) {
      if (shear_fold) {
        std::cout << "# dust: shear-periodic x1 boundaries active; ghost deposits are "
                  << "folded across the radial faces with a conservative y-remap "
                  << "(<dust>/shear_remap = " << rmap << ")" << std::endl;
      } else {
        std::cout << "# WARNING (dust): <dust>/shear_fold = false: ghost deposits are "
                  << "NOT folded across the shear-periodic radial faces (diagnostic "
                  << "mode; the x1-face deposits of the face blocks are dropped)"
                  << std::endl;
      }
    }
  }
  if (pmy_pack->pmesh->mb_indcs.ng < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust drag requires at least 2 ghost zones" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) read parameters ----------------------------------------------------------------
  back_reaction = pin->GetOrAddBoolean("dust","back_reaction",true);
  gamma_switch  = pin->GetOrAddBoolean("dust","gamma_switch",false);
  if (coupling == DustCoupling::hybrid && gamma_switch) {
    if (global_variable::my_rank == 0) {
      std::cout << "# WARNING (dust): <dust>/gamma_switch=true has no effect for "
                << "coupling=" << (pc2_only ? "pc2" : "hybrid")
                << "; gamma_switch only modifies the imex2+ tableau and will be ignored."
                << std::endl;
    }
    gamma_switch = false;
  }
  stopping_times_initialized = false;
  dt_cfl        = pin->GetOrAddReal("dust","dt_cfl",0.5);
  dust_to_gas   = pin->GetOrAddReal("dust","dust_to_gas",0.01);

  // particle transport timestep: which azimuthal velocity sets the cell-crossing limit,
  // and how much of a MeshBlock a particle may cross in one stage (see DustDtTransport)
  {
    std::string dtt = pin->GetOrAddString("dust","dt_transport","relative");
    if (dtt.compare("relative") == 0) {
      dt_transport = DustDtTransport::relative;
    } else if (dtt.compare("full") == 0) {
      dt_transport = DustDtTransport::full;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/dt_transport = '" << dtt
                << "' not implemented. Valid choices are [relative,full]." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  dt_block_safety = pin->GetOrAddReal("dust","dt_block_safety",0.5);
  if (!(dt_block_safety > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/dt_block_safety = " << dt_block_safety << " must be positive"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (dt_block_safety >= 1.0 && global_variable::my_rank == 0) {
    std::cout << "# WARNING (dust): <dust>/dt_block_safety = " << dt_block_safety
              << " >= 1 lets a particle cross a whole MeshBlock in one stage, which the "
              << "single-hop particle migration cannot resolve; expect the PM halo "
              << "abort once the shear is fast enough.  Diagnostic use only."
              << std::endl;
  }

  hybrid_enter_zeta = pin->GetOrAddReal("dust", "hybrid_enter_zeta", 0.5);
  hybrid_enter_chi  = pin->GetOrAddReal("dust", "hybrid_enter_chi", 0.5);
  hybrid_exit_zeta  = pin->GetOrAddReal("dust", "hybrid_exit_zeta", 0.25);
  hybrid_exit_chi   = pin->GetOrAddReal("dust", "hybrid_exit_chi", 0.25);
  if (!(hybrid_enter_zeta > hybrid_exit_zeta && hybrid_exit_zeta >= 0.0 &&
        hybrid_enter_chi > hybrid_exit_chi && hybrid_exit_chi >= 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Hybrid enter thresholds must be positive and exceed exit thresholds"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pc2_only) {
    hybrid_force_mode = HybridForceMode::pc2;
    hybrid_mode = HybridMode::pc2;
  } else {
    std::string forced = pin->GetOrAddString("dust", "hybrid_force_mode", "auto");
    if (forced.compare("auto") == 0) {
      hybrid_force_mode = HybridForceMode::automatic;
      hybrid_mode = HybridMode::pc2;
    } else if (forced.compare("pc2") == 0) {
      hybrid_force_mode = HybridForceMode::pc2;
      hybrid_mode = HybridMode::pc2;
    } else if (forced.compare("split_be") == 0) {
      hybrid_force_mode = HybridForceMode::split_be;
      hybrid_mode = HybridMode::split_be;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/hybrid_force_mode = '" << forced
                << "' not recognized (must be auto, pc2, or split_be)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  {
    std::string solver = pin->GetOrAddString("dust", "drag_solver", "local");
    if (solver.compare("local") == 0) {
      drag_solver = DustDragSolver::local;
    } else if (solver.compare("applya") == 0) {
      drag_solver = DustDragSolver::applya;
    } else if (solver.compare("dc1") == 0) {
      drag_solver = DustDragSolver::dc1;
    } else if (solver.compare("dc2") == 0) {
      drag_solver = DustDragSolver::dc2;
    } else if (solver.compare("pcg") == 0) {
      drag_solver = DustDragSolver::pcg;
    } else if (solver.compare("adaptive") == 0) {
      drag_solver = DustDragSolver::adaptive;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/drag_solver = '" << solver
                << "' not recognized (must be local, applya, dc1, dc2, pcg, or adaptive)"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (coupling == DustCoupling::hybrid &&
      drag_solver != DustDragSolver::local && drag_solver != DustDragSolver::dc1 &&
      drag_solver != DustDragSolver::pcg) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "The first hybrid split-BE implementation supports only "
              << "drag_solver=local, dc1, or pcg" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  drag_rtol = pin->GetOrAddReal("dust", "drag_rtol", 1.0e-11);
  drag_atol = pin->GetOrAddReal("dust", "drag_atol", 1.0e-14);
  drag_iter_max = pin->GetOrAddInteger("dust", "drag_iter_max", 200);
  drag_diagnostic_interval = pin->GetOrAddInteger("dust", "drag_diagnostic_interval", 0);
  adaptive_order_c = pin->GetOrAddReal("dust", "drag_adaptive_order_c", 0.25);
  adaptive_rtol_max = pin->GetOrAddReal("dust", "drag_adaptive_rtol_max", 1.0e-3);
  adaptive_tref = pin->GetOrAddReal("dust", "drag_adaptive_tref", 1.0);
  adaptive_state_floor = pin->GetOrAddReal("dust", "drag_adaptive_state_floor", 1.0);
  std::string fail_policy = pin->GetOrAddString("dust", "drag_fail_policy", "abort");
  if (drag_rtol <= 0.0 || drag_atol < 0.0 || drag_iter_max < 1 ||
      drag_diagnostic_interval < 0 || adaptive_order_c <= 0.0 ||
      adaptive_rtol_max <= 0.0 || adaptive_tref <= 0.0 ||
      adaptive_state_floor <= 0.0 || fail_policy.compare("abort") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid coupled drag-solver tolerance/iteration policy; "
              << "only drag_fail_policy=abort is supported" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0 &&
      (drag_solver == DustDragSolver::pcg || drag_solver == DustDragSolver::adaptive)) {
    solver_pcg_iteration_hist.assign(static_cast<std::size_t>(drag_iter_max) + 1, 0);
  }

  {
    std::string mode = pin->GetOrAddString("dust","stopping_time_mode","species_fixed");
    if (mode.compare("species_fixed") == 0) {
      stopping_time_mode = DustStoppingTimeMode::species_fixed;
    } else if (mode.compare("particle_static") == 0) {
      stopping_time_mode = DustStoppingTimeMode::particle_static;
    } else if (mode.compare("dynamic") == 0) {
      stopping_time_mode = DustStoppingTimeMode::dynamic;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/stopping_time_mode = '" << mode
                << "' not recognized (must be species_fixed, particle_static, or "
                << "dynamic)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // drag work is not deposited into the gas energy equation, so back-reaction requires
  // Ideal gas: every drag kick on the gas momentum is applied with the matching change
  // of the kinetic energy (internal energy unchanged); <dust>/drag_heating additionally
  // deposits the frictional dissipation Q_j = m_j c_j (1 - c_j/2) |u~ - v_j|^2 of each
  // particle kick as heat, which conserves the total gas + dust energy.
  gas_ideal = phyd->peos->eos_data.is_ideal;
  drag_heating = pin->GetOrAddBoolean("dust","drag_heating",false);
  if (drag_heating && !gas_ideal) {
    if (global_variable::my_rank == 0) {
      std::cout << "# WARNING (dust): <dust>/drag_heating = true has no effect with an "
                << "isothermal EOS; ignored." << std::endl;
    }
    drag_heating = false;
  }
  if (back_reaction && gas_ideal && global_variable::my_rank == 0) {
    std::cout << "# dust: ideal-gas EOS with back-reaction: drag kicks keep the gas "
              << "internal energy fixed" << (drag_heating ?
              "; frictional dissipation deposited as heat (<dust>/drag_heating = true)" :
              " (<dust>/drag_heating = false: dissipation discarded)") << std::endl;
  }

  // per-species stopping times: <dust>/nspecies and taus_1, taus_2, ...
  nspecies = pin->GetOrAddInteger("dust","nspecies",1);
  if (nspecies < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/nspecies must be at least 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  taus = DualArray1D<Real>("dust_taus", nspecies);
  taus_max = 0.0;
  taus_min = std::numeric_limits<Real>::max();
  for (int s=0; s<nspecies; ++s) {
    Real ts = pin->GetReal("dust", "taus_" + std::to_string(s+1));
    if (ts <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dust stopping times must be positive" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    taus.h_view(s) = ts;
    taus_max = std::max(taus_max, ts);
    taus_min = std::min(taus_min, ts);
  }
  taus.template modify<HostMemSpace>();
  taus.template sync<DevExeSpace>();

  // deposit scheme
  {
    std::string dep = pin->GetOrAddString("dust","deposit","tsc");
    if (dep.compare("ngp") == 0) {
      deposit = DustDeposit::ngp;
    } else if (dep.compare("cic") == 0) {
      deposit = DustDeposit::cic;
    } else if (dep.compare("tsc") == 0) {
      deposit = DustDeposit::tsc;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/deposit = '" << dep << "' not recognized "
                << "(must be ngp, cic, or tsc)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // shearing box parameters (must match those read by the ShearingBox constructor)
  is_shearing_box = pin->DoesBlockExist("shearing_box");
  if (is_shearing_box) {
    qshear = pin->GetReal("shearing_box","qshear");
    omega0 = pin->GetReal("shearing_box","omega0");
    is_stratified = pin->GetOrAddBoolean("shearing_box","stratified",false);
  } else {
    qshear = 0.0;
    omega0 = 0.0;
    is_stratified = false;
  }
  if (drag_solver != DustDragSolver::local && shear_x1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Coupled drag solvers require ordinary periodic boundaries; "
              << "the exact-transpose shear deposit is not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // (2c) self-gravity coupling (Phase 4c): the dust mass density joins the Poisson
  // source and the particles feel the gradient of the total potential
  {
    bool have_grav = (pmy_pack->pgrav != nullptr);
    gravity = pin->GetOrAddBoolean("dust","gravity",have_grav);
    gravity_source = pin->GetOrAddBoolean("dust","gravity_source",true);
    gravity_force  = pin->GetOrAddBoolean("dust","gravity_force",true);
    if (gravity && !have_grav) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/gravity = true requires a <gravity> block"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (gravity && !(pmy_pack->pmesh->three_d)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dust self-gravity requires a 3D mesh" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (gravity && !gravity_source && !gravity_force) {gravity = false;}
    if (global_variable::my_rank == 0) {
      if (gravity) {
        std::cout << "# dust: self-gravity coupling on: "
                  << (gravity_source ? "dust density in the Poisson source" :
                                       "dust NOT in the Poisson source")
                  << ", particles " << (gravity_force ? "feel" : "do not feel")
                  << " -grad(phi)" << std::endl;
      } else if (have_grav) {
        std::cout << "# dust: <gravity> present but <dust>/gravity = false: dust is "
                  << "neither a source of nor subject to self-gravity"
                  << std::endl;
      }
    }
  }
  if (is_shearing_box && pmy_pack->pmesh->three_d &&
      global_variable::my_rank == 0) {
    if (dt_transport == DustDtTransport::relative) {
      std::cout << "# dust: particle transport timestep uses the shear-subtracted "
                << "azimuthal velocity; the background shear is limited instead to "
                << dt_block_safety << " of a MeshBlock width per stage "
                << "(<dust>/dt_transport = relative)" << std::endl;
    } else {
      std::cout << "# dust: particle transport timestep uses the full azimuthal "
                << "velocity v_y - q*Omega*x (<dust>/dt_transport = full)" << std::endl;
    }
  }

  // (3) allocate deposited fields (with ghost zones) ------------------------------------
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  // the PM dust density is always available (the dust_dpm output, the Poisson source)
  Kokkos::realloc(rho_dust, nmb, 1, ncells3, ncells2, ncells1);
  if (gravity) {
    Kokkos::realloc(gforce,   nmb, 3, ncells3, ncells2, ncells1);
  }
  Kokkos::realloc(qdep,  nmb, 5, ncells3, ncells2, ncells1);
  Kokkos::realloc(ustar, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(dmom,  nmb, 4, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(dmom, 0.0);  // read as R_g=0 in stage 2 if back_reaction is off
  if (drag_solver == DustDragSolver::pcg || drag_solver == DustDragSolver::adaptive) {
    Kokkos::realloc(solver_r, nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(solver_p, nmb, 3, ncells3, ncells2, ncells1);
  }
  if (drag_solver != DustDragSolver::local) {
    Kokkos::realloc(solver_ap, nmb, 3, ncells3, ncells2, ncells1);
  }

  // (4) allocate boundary communication objects -----------------------------------------
  pbval_qp = new MeshBoundaryValuesDep(pmy_pack, pin);
  pbval_qp->InitializeBuffers(5);
  pbval_dm = new MeshBoundaryValuesDep(pmy_pack, pin);
  pbval_dm->InitializeBuffers(4);
  pbval_us = new MeshBoundaryValuesCC(pmy_pack, pin, false);
  pbval_us->InitializeBuffers(3);
  if (drag_solver != DustDragSolver::local) {
    // These objects own distinct communicators/requests and are reinitialized for every
    // matrix-free matvec; they must not alias the one-shot stage exchanges above.
    pbval_solver_copy = new MeshBoundaryValuesCC(pmy_pack, pin, false);
    pbval_solver_copy->InitializeBuffers(3);
    pbval_solver_add = new MeshBoundaryValuesDep(pmy_pack, pin);
    pbval_solver_add->InitializeBuffers(3);
  }
  // shear-periodic remap of the u* radial ghost zones (3D shearing box only)
  if (shear_x1 && pmy_pack->pmesh->three_d) {
    psbox_us = new ShearingBoxCC(pmy_pack, pin, 3);
  }
  // self-gravity: the dust density (additive exchange + copy exchange of its ghost
  // layer, both with the shear remap) and the force field (copy exchange)
  pbval_rd = new MeshBoundaryValuesDep(pmy_pack, pin);
  pbval_rd->InitializeBuffers(1);
  pbval_rc = new MeshBoundaryValuesCC(pmy_pack, pin, false);
  pbval_rc->InitializeBuffers(1);
  if (shear_x1) {psbox_rc = new ShearingBoxCC(pmy_pack, pin, 1);}
  if (gravity && gravity_source) {
    pmy_pack->pgrav->RegisterExtraDensity(rho_dust);
  }
  if (gravity && gravity_force) {
    pbval_g = new MeshBoundaryValuesCC(pmy_pack, pin, false);
    pbval_g->InitializeBuffers(3);
    if (shear_x1) {psbox_g = new ShearingBoxCC(pmy_pack, pin, 3);}
  }
}

//----------------------------------------------------------------------------------------
// destructor

DustGasDrag::~DustGasDrag() {
  if (physical_faces) {
    unsigned long long ntot = escaped_count;
    Real mtot = escaped_mass;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &ntot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mtot, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << std::setprecision(14) << "# DUST_ESCAPE_SUMMARY removed=" << ntot
                << " mass=" << mtot << std::endl;
    }
  }
  if (global_variable::my_rank == 0 && dt_guard_count > 0) {
    // rank-local count: the guard may bind on another rank without appearing here, and
    // the timestep actually taken is the global minimum over ranks
    std::cout << "# DUST_DT_SUMMARY block_guard_cycles=" << dt_guard_count
              << " (rank 0; <dust>/dt_block_safety = " << dt_block_safety << ")"
              << std::endl;
  }
  if (global_variable::my_rank == 0 && coupling == DustCoupling::hybrid) {
    unsigned long long total = hybrid_pc2_cycles + hybrid_split_be_cycles;
    double pc2_fraction = (total > 0) ?
        static_cast<double>(hybrid_pc2_cycles)/static_cast<double>(total) : 0.0;
    std::cout << std::setprecision(14)
              << "# DUST_HYBRID_SUMMARY pc2_cycles=" << hybrid_pc2_cycles
              << " split_be_cycles=" << hybrid_split_be_cycles
              << " pc2_fraction=" << pc2_fraction
              << " last_zeta=" << hybrid_last_zeta
              << " last_chi=" << hybrid_last_chi << std::endl;
  }
  if (global_variable::my_rank == 0 && solver_stage_count > 0) {
    auto quantile = [&](double fraction) {
      if (solver_pcg_stage_count == 0) return 0;
      unsigned long long order = (fraction <= 0.0) ? 1 :
          static_cast<unsigned long long>(
              std::ceil(fraction*static_cast<double>(solver_pcg_stage_count)));
      unsigned long long cumulative = 0;
      for (std::size_t iteration=0; iteration<solver_pcg_iteration_hist.size();
           ++iteration) {
        cumulative += solver_pcg_iteration_hist[iteration];
        if (cumulative >= order) return static_cast<int>(iteration);
      }
      // The histogram and stage count are updated together; reaching this return would
      // indicate internal diagnostic corruption rather than a numerical solver failure.
      return drag_iter_max;
    };
    const char *name = (drag_solver == DustDragSolver::applya) ? "applya" :
                       (drag_solver == DustDragSolver::dc1) ? "dc1" :
                       (drag_solver == DustDragSolver::dc2) ? "dc2" :
                       (drag_solver == DustDragSolver::pcg) ? "pcg" :
                       (drag_solver == DustDragSolver::adaptive) ? "adaptive" : "local";
    std::cout << std::setprecision(14)
              << "# DUST_SOLVER_SUMMARY mode=" << name
              << " stages=" << solver_stage_count
              << " applya=" << solver_applya_count
              << " halos=" << solver_halo_count
              << " reductions=" << solver_reduction_count
              << " fast_accept=" << solver_fast_accept_count
              << " pcg_stages=" << solver_pcg_stage_count
              << " pcg_iter_min=" << quantile(0.0)
              << " pcg_iter_median=" << quantile(0.5)
              << " pcg_iter_p95=" << quantile(0.95)
              << " pcg_iter_max=" << quantile(1.0)
              << " solve_seconds=" << solver_wall_seconds << std::endl;
  }
  delete pbval_qp;
  delete pbval_dm;
  delete pbval_us;
  if (pbval_solver_copy != nullptr) delete pbval_solver_copy;
  if (pbval_solver_add != nullptr) delete pbval_solver_add;
  if (psbox_us != nullptr) {delete psbox_us;}
  if (pbval_rd != nullptr) {delete pbval_rd;}
  if (pbval_rc != nullptr) {delete pbval_rc;}
  if (psbox_rc != nullptr) {delete psbox_rc;}
  if (pbval_g  != nullptr) {delete pbval_g;}
  if (psbox_g  != nullptr) {delete psbox_g;}
}

//----------------------------------------------------------------------------------------
//! \fn bool DustGasDrag::ActiveStage
//! \brief Returns false on stages where all dust work is dormant. For imex2+ the final
//! explicit stage is a trivial assembly step (beta=0, gam0=1, gam1=0, and the implicit
//! solve is skipped since estage == nexp_stages), so every dust task no-ops there.

bool DustGasDrag::ActiveStage(Driver *pdrive, int stage) const {
  if (coupling == DustCoupling::hybrid) return true;
  return !((pdrive->integrator.compare("imex2+") == 0) && (stage == pdrive->nexp_stages));
}

bool DustGasDrag::FirstMigrationActive(Driver *pdrive, int stage) const {
  if (coupling == DustCoupling::imex) return ActiveStage(pdrive, stage);
  return (hybrid_mode == HybridMode::pc2 && stage == 2);
}

bool DustGasDrag::SecondMigrationActive(Driver *pdrive, int stage) const {
  return (coupling == DustCoupling::hybrid &&
          hybrid_mode == HybridMode::split_be && stage == 2);
}

Real DustGasDrag::DragStep(Driver *pdrive) const {
  if (coupling == DustCoupling::hybrid) return pmy_pack->pmesh->dt;
  return (pdrive->a_impl)*(pmy_pack->pmesh->dt);
}

//----------------------------------------------------------------------------------------
//! \fn void DustGasDrag::SetDefaultMasses
//! \brief Default per-particle species/stopping-time/mass initialization: species are
//! striped over the particle index, and masses are normalized so the total dust mass is
//! dust_to_gas times the gas mass for a uniform gas of density <problem>/rho0.
//! Problem generators may overwrite PSP/IPTS/IPM after calling this.

void DustGasDrag::SetDefaultMasses(ParameterInput *pin) {
  particles::Particles *ppar = pmy_pack->ppart;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &mbsize = pmy_pack->pmb->mb_size;
  bool three_d = pmy_pack->pmesh->three_d;
  Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &taus_ = taus;
  auto gids = pmy_pack->gids;
  int nspec = nspecies;
  Real d2g = dust_to_gas;
  par_for("dust_masses",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) {vol *= mbsize.d_view(m).dx3;}
    int s = p % nspec;
    pi(PSP,p) = s;
    pr(IPTS,p) = taus_.d_view(s);
    pr(IPM,p) = d2g*rho0*vol/ppc;
  });
}

} // namespace dust
