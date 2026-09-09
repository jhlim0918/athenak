#ifndef DUST_DUST_HPP_
#define DUST_DUST_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust.hpp
//! \brief definitions for DustGasDrag class: Lagrangian dust particles coupled to a
//! Hydro gas by stiff mutual drag.  The module supports the imex2+ IMEX(4,3,2) method
//! (Krapp et al. 2024), plus an RK2 hybrid of resolved PC2 midpoint coupling and a stiff
//! full-step split-BE fallback.  Both use the particle-mesh scatter/gather method of
//! Yang & Johansen (2016) with exact momentum-conserving back-reaction (PMBR), and the
//! O(N) closed-form per-cell drag solve of Benitez-Llambay, Krapp & Pessah (2019)
//! generalized to per-particle stopping times.
//!
//! This module couples the Particles and Hydro modules the same way IonNeutral couples
//! Hydro and MHD: it owns the combined task graph, the deposited fields, and the drag
//! kernels, while particle storage/migration stays in Particles and fluxes/BCs in Hydro.

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class Driver;
class MeshBlockPack;
class ShearingBoxCC;

// constants that enumerate particle-mesh deposit schemes
enum class DustDeposit {ngp=0, cic=1, tsc=2};

// contract for per-particle stopping times.  The default species_fixed mode keeps the
// configured species table authoritative; particle_static permits arbitrary positive
// per-particle values that remain constant; dynamic refreshes their global maximum once
// per cycle.
enum class DustStoppingTimeMode {species_fixed=0, particle_static=1, dynamic=2};

// `local` is the regression baseline. `applya` executes one non-mutating coupled
// operator application and then retains the local answer, for marginal-cost timing.
// `dc1` and `dc2` unconditionally accept one and two fixed defect corrections.
enum class DustDragSolver {local=0, applya=1, dc1=2, dc2=3, pcg=4, adaptive=5};

// Time coupling between the hydro RK update and particle drag. Input coupling=pc2 maps
// to the hybrid task path with PC2 forced for every cycle. Input coupling=hybrid can use
// PC2 while drag and feedback are resolved and split backward-Euler otherwise.
enum class DustCoupling {imex=0, hybrid=1};

//----------------------------------------------------------------------------------------
//! \brief Velocity used for the particle transport timestep in the 3D shearing box.
//! full     = dx2/|v_y - q*Omega*x| (legacy): one cell per step of the *total* azimuthal
//!            transport, so the timestep collapses like 1/Lx in a wide box.
//! relative = dx2/|v_y| : the background shear is excluded, mirroring the orbital
//!            advection of the gas.  The MeshBlock-width guard below keeps the residual
//!            shear displacement addressable by the (single-hop) particle migration.
enum class DustDtTransport {full=0, relative=1};
enum class HybridMode {pc2=0, split_be=1};
enum class HybridForceMode {automatic=0, pc2=1, split_be=2};

//----------------------------------------------------------------------------------------
//! \fn void GasKick
//! \brief Add a momentum kick (dmx, dmy, dmz) to the gas conserved variables of one cell
//! and, for an ideal gas, the matching change of the kinetic energy so that the internal
//! energy is unchanged by the kick.  Every drag-induced change of the gas momentum in the
//! dust module goes through this helper (back-reaction apply, IMEX history term, PC2
//! predictor apply and removal); frictional heating, when enabled, is added separately.

KOKKOS_INLINE_FUNCTION
void GasKick(const DvceArray5D<Real> &u0, const int m, const int k, const int j,
             const int i, const Real dmx, const Real dmy, const Real dmz,
             const bool ideal) {
  if (ideal) {
    const Real irho = 1.0/u0(m,IDN,k,j,i);
    const Real m1 = u0(m,IM1,k,j,i), m2 = u0(m,IM2,k,j,i), m3 = u0(m,IM3,k,j,i);
    const Real n1 = m1 + dmx, n2 = m2 + dmy, n3 = m3 + dmz;
    u0(m,IEN,k,j,i) += 0.5*irho*((n1*n1 + n2*n2 + n3*n3) - (m1*m1 + m2*m2 + m3*m3));
  }
  u0(m,IM1,k,j,i) += dmx;
  u0(m,IM2,k,j,i) += dmy;
  u0(m,IM3,k,j,i) += dmz;
}

//----------------------------------------------------------------------------------------
//! \struct DustGasDragTaskIDs
//  \brief container to hold TaskIDs of all dust+hydro tasks

struct DustGasDragTaskIDs {
  // "before_timeintegrator" tasks
  TaskID gswitch;
  // "before_stagen" tasks
  TaskID h_irecv, irecvd;
  // "stagen" tasks
  TaskID first2;
  TaskID h_flux, h_sendf, h_recvf, h_rkupdt, h_srctrms;
  TaskID gatwid, push;
  TaskID p_newgid, p_cnt, p_irecv, p_sendp, p_recvp;
  TaskID scat, sendd, recvd, foldd, solve, sendus, recvus, sendus_shr, recvus_shr;
  TaskID gkp, sendbr, recvbr, foldbr, apply;
  TaskID p2_newgid, p2_cnt, p2_irecv, p2_sendp, p2_recvp;
  TaskID h_sendu_oa, h_recvu_oa, h_restu, h_sendu, h_recvu, h_sendu_shr, h_recvu_shr;
  TaskID h_bcs, h_prol, h_c2p, h_newdt, newdt, newdt2;
  // "after_stagen" tasks
  TaskID h_csend, h_crecv, p_csend, p_crecv, cleard;
  TaskID p_remove, p2_remove;
  // self-gravity (Phase 4c): dust density into the Poisson source, force on particles
  TaskID gdep, gsend, grecv, gfold, gsendc, grecvc, gsends, grecvs;
  TaskID gforce, gsendf, grecvf, gsendfs, grecvfs;
  // SMR: prolongation of the copy-exchanged fields at fine/coarse boundaries
  TaskID prolus, gprolc, gprolf;
};

namespace dust {

// Particle deposits need atomic updates on parallel execution spaces because stencil
// footprints overlap.  Kokkos::Serial has only one worker, however, so its generic
// compare-and-swap atomic adds synchronization overhead without providing protection.
// Keep the operation order unchanged while compiling that overhead out of serial CPU
// builds; all genuinely parallel backends retain Kokkos atomics.
KOKKOS_INLINE_FUNCTION
void DepositAdd(Real *destination, const Real value) {
#if defined(KOKKOS_ENABLE_SERIAL)
  constexpr bool serial_exec = std::is_same_v<DevExeSpace, Kokkos::Serial>;
#else
  constexpr bool serial_exec = false;
#endif
  if constexpr (serial_exec) {
    *destination += value;
  } else {
    Kokkos::atomic_add(destination, value);
  }
}

// Shared NGP/CIC/TSC stencil for deposits, matrix-free gathers, and the final kick.
// One helper is used everywhere so the coupled operator and conservative commit cannot
// silently acquire different particle-mesh weights.
KOKKOS_INLINE_FUNCTION
void PMWeights(const Real x, const Real xmin, const Real xmax, const int nx,
               const int is, const int scheme, int &ip, Real w[3]) {
  // Express the particle position directly in cell coordinates.  Besides being the
  // same uniform-grid geometry as CellCenterX, this uses one division rather than
  // forming dx and then dividing by it twice.
  Real cell = (x - xmin)*(static_cast<Real>(nx)/(xmax - xmin));
  int ig = static_cast<int>(cell + 1.0) - 1;
  Real del = cell - (static_cast<Real>(ig) + 0.5);
  ip = ig + is;
  if (scheme == 0) {
    w[0] = 0.0;
    w[1] = 1.0;
    w[2] = 0.0;
  } else if (scheme == 1) {
    w[0] = fmax(0.0, -del);
    w[1] = 1.0 - fabs(del);
    w[2] = fmax(0.0, del);
  } else {
    w[0] = 0.5*SQR(0.5 - del);
    w[1] = 0.75 - SQR(del);
    w[2] = 0.5*SQR(0.5 + del);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DepositStencil
//! \brief Particle-mesh stencil used for every DEPOSIT (scatter) of the module.  With
//! static mesh refinement all MeshBlocks deposit with the kernel of the FINEST level so
//! that the deposited density is one consistent field across level boundaries: a block
//! one level below the finest (r = 2) scatters into its fine image (r*nx cells and r*ng
//! ghost cells per direction, cell volume V/r^d), which is then volume-averaged onto its
//! own cells (DustGasDrag::RestrictImage); a block at the finest level (r = 1) scatters
//! into its own cells.  Gathers keep using the block's own cells (PMWeights directly).
//! On a uniform mesh r = 1 and the arithmetic is unchanged.

KOKKOS_INLINE_FUNCTION
void DepositStencil(const RegionSize &sz, const Real x, const Real y, const Real z,
                    const int nx1, const int nx2, const int nx3,
                    const int is, const int js, const int ks, const int r,
                    const bool three_d, const int scheme,
                    int &ip, int &jp, int &kp, Real wx[3], Real wy[3], Real wz[3],
                    Real &vol) {
  PMWeights(x, sz.x1min, sz.x1max, r*nx1, r*is, scheme, ip, wx);
  PMWeights(y, sz.x2min, sz.x2max, r*nx2, r*js, scheme, jp, wy);
  vol = sz.dx1*sz.dx2;
  if (three_d) {
    PMWeights(z, sz.x3min, sz.x3max, r*nx3, r*ks, scheme, kp, wz);
    vol *= sz.dx3;
    if (r > 1) {vol /= static_cast<Real>(r*r*r);}
  } else {
    kp = ks;
    wz[0] = 0.0; wz[1] = 1.0; wz[2] = 0.0;
    if (r > 1) {vol /= static_cast<Real>(r*r);}
  }
}

//----------------------------------------------------------------------------------------
//! \class DustGasDrag

class DustGasDrag {
 public:
  DustGasDrag(MeshBlockPack *ppack, ParameterInput *pin);
  ~DustGasDrag();

  // data
  int nspecies;              // number of dust species (per-species stopping times)
  bool back_reaction;        // dust exerts drag on gas (PMBR); false = test particles
  bool gamma_switch;         // switch gamma to 1/2 when dt > max stopping time (Krapp24)
  bool stopping_times_initialized;  // post-pgen stopping-time contract has been checked
  bool is_shearing_box;      // <shearing_box> block present
  bool is_stratified;        // vertical gravity (3D shearing box only)
  Real qshear, omega0;       // shearing box parameters (0 if no shearing box)
  Real dt_cfl;               // particle CFL number for transport timestep
  DustDtTransport dt_transport;  // azimuthal velocity used for the cell-crossing limit
  Real dt_block_safety;      // fraction of a MeshBlock crossed per stage (guard)
  unsigned long long dt_guard_count = 0;   // cycles where the block guard set dtnew
  bool physical_faces = false;      // some mesh face is non-periodic: particles can leave
  unsigned long long escaped_count = 0;    // particles removed through physical faces
  Real escaped_mass = 0.0;                 // their total mass (this rank)
  Real taus_max;             // largest stopping time over all species
  Real taus_min;             // smallest stopping time over all species
  Real dust_to_gas;          // total dust/gas mass ratio for default mass normalization
  DualArray1D<Real> taus;    // per-species stopping times
  DustDeposit deposit;       // particle-mesh deposit scheme (tsc default)
  ReconstructionMethod shear_remap;  // y-remap of the shear-periodic deposit fold (plm)
  bool shear_fold;           // false = diagnostic: skip the fold (unsheared x1 deposits)
  bool gas_ideal;            // ideal-gas EOS: drag kicks carry an energy update
  bool drag_heating;         // ideal gas: deposit the frictional dissipation as heat
  bool gravity;              // dust takes part in self-gravity (<gravity> block, 3D)
  bool gravity_source;       // the PM dust density is added to the Poisson source
  bool gravity_force;        // particles feel -grad(phi) of the total potential
  DustStoppingTimeMode stopping_time_mode;
  DustDragSolver drag_solver;
  DustCoupling coupling;
  HybridMode hybrid_mode;
  HybridForceMode hybrid_force_mode;
  Real hybrid_enter_zeta, hybrid_enter_chi;
  Real hybrid_exit_zeta, hybrid_exit_chi;
  Real hybrid_last_zeta = 0.0;
  Real hybrid_last_chi = 0.0;
  Real hybrid_feedback_rate_max = 0.0;
  bool hybrid_have_metric = false;
  unsigned long long hybrid_pc2_cycles = 0;
  unsigned long long hybrid_split_be_cycles = 0;
  Real drag_rtol, drag_atol;
  int drag_iter_max, drag_diagnostic_interval;
  Real adaptive_order_c, adaptive_rtol_max, adaptive_tref, adaptive_state_floor;

  // deposited fields, dimensioned (nmb, nvar, ncells3, ncells2, ncells1)
  DvceArray5D<Real> qdep;    // [0]=Q, [1-3]=P/predictor impulse, [4]=feedback rate
  DvceArray5D<Real> ustar;   // nvar=3: provisional drag-corrected gas velocity u*
  DvceArray5D<Real> dmom;    // nvar=4: PMBR momentum deposit [0-2] and frictional heat
  DvceArray5D<Real> rho_dust;   // nvar=1: PM dust mass density (Poisson source, 4c)
  DvceArray5D<Real> gforce;     // nvar=3: -grad(phi) on the mesh, ghost-filled (4c)
                             // [3] (ideal gas with drag_heating); becomes R_g after apply
  DvceArray5D<Real> cdummy;  // 1-element dummy coarse array (coupled-solver exchanges)
  // SMR: restricted copies of the copy-exchanged fields, (nmb, nvar, cnx3+2ng, ...);
  // 1-element dummies on a uniform mesh (PackAndSendCC never reads them then)
  DvceArray5D<Real> coarse_us;  // u*
  DvceArray5D<Real> coarse_rd;  // rho_dust
  DvceArray5D<Real> coarse_g;   // gforce
  bool multilevel;           // SMR mesh: restrict/prolongate around the copy exchanges
  // SMR: finest-level deposits.  rfac(m) = 2 on blocks one level below the finest level
  // (they scatter into the fine images below, then RestrictImage averages onto their
  // cells), 1 otherwise.  any_coarse = some block of this pack has rfac = 2.
  bool any_coarse = false;
  DualArray1D<int> rfac;
  DvceArray5D<Real> fimg_q;     // fine image of qdep (5 vars)
  DvceArray5D<Real> fimg_d;     // fine image of dmom (4 vars)
  DvceArray5D<Real> fimg_r;     // fine image of rho_dust (1 var)
  DvceArray5D<Real> solver_r;   // coupled-solver residual
  DvceArray5D<Real> solver_p;   // PCG search direction
  DvceArray5D<Real> solver_ap;  // matrix-free A*x / A*p work field

  // Boundary communication objects
  MeshBoundaryValuesDep *pbval_qp;  // additive exchange of (Q,P) ghost deposits
  MeshBoundaryValuesDep *pbval_dm;  // additive exchange of PMBR ghost deposits
  MeshBoundaryValuesCC  *pbval_us;  // copy exchange to fill u* ghost zones
  ShearingBoxCC *psbox_us = nullptr;  // shear-periodic remap of u* x1 ghost zones (3D)
  MeshBoundaryValuesDep *pbval_rd = nullptr;  // additive exchange of rho_dust deposits
  MeshBoundaryValuesCC  *pbval_rc = nullptr;  // copy exchange filling rho_dust ghosts
  ShearingBoxCC *psbox_rc = nullptr;          // shear remap of the rho_dust x1 ghosts
  MeshBoundaryValuesCC  *pbval_g = nullptr;   // copy exchange filling gforce ghosts
  ShearingBoxCC *psbox_g = nullptr;           // shear remap of the gforce x1 ghosts
  MeshBoundaryValuesCC *pbval_solver_copy = nullptr;
  MeshBoundaryValuesDep *pbval_solver_add = nullptr;

  // Cumulative rank-local timing and globally consistent iteration diagnostics.
  unsigned long long solver_stage_count = 0;
  unsigned long long solver_applya_count = 0;
  unsigned long long solver_halo_count = 0;
  unsigned long long solver_reduction_count = 0;
  unsigned long long solver_fast_accept_count = 0;
  unsigned long long solver_pcg_stage_count = 0;
  double solver_wall_seconds = 0.0;
  // Exact bounded histogram over [0, drag_iter_max], allocated only on rank 0.
  std::vector<unsigned long long> solver_pcg_iteration_hist;
  Real solver_last_residual = -1.0;  // negative means a true residual was not sampled
  Real solver_last_epsmax = -1.0;  // negative means this mode did not sample epsilon_c
  Real solver_last_error_bound = -1.0;
  Real solver_last_acceptance_target = -1.0;
  Real solver_last_state_scale = -1.0;
  int solver_last_iterations = 0;
  bool solver_last_fast_accept = false;

  // container to hold names of TaskIDs
  DustGasDragTaskIDs id;

  // functions...
  void AssembleDustGasDragTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // dust tasks are dormant on the trivial assembly stage of imex2+ (beta=0, gam0=1)
  bool ActiveStage(Driver *pdrive, int stage) const;
  bool FirstMigrationActive(Driver *pdrive, int stage) const;
  bool SecondMigrationActive(Driver *pdrive, int stage) const;
  Real DragStep(Driver *pdrive) const;
  // default particle mass/species/stopping-time initialization (pgen may override)
  void SetDefaultMasses(ParameterInput *pin);

  // ...in "before_timeintegrator" list
  TaskStatus GammaSwitch(Driver *pdrive, int stage);
  // Must be public because NVCC extended host/device lambdas cannot be enclosed by a
  // private or protected member function.
  void RefreshStoppingTimeMaximum();  // validate IPTS; refresh taus_max when requested
  // ...in "before_stagen" list
  TaskStatus InitRecvDep(Driver *pdrive, int stage);
  // ...in "stagen" list
  TaskStatus FirstTwoImpRK(Driver *pdrive, int stage);     // stage 1 register copies
  TaskStatus ExplicitPush(Driver *pdrive, int stage);      // rotation kick + drift
  TaskStatus UpdateParticleGIDs(Driver *pdrive, int stage);
  TaskStatus CountParticleSends(Driver *pdrive, int stage);
  TaskStatus InitParticleRecv(Driver *pdrive, int stage);
  TaskStatus SendParticles(Driver *pdrive, int stage);
  TaskStatus RecvParticles(Driver *pdrive, int stage);
  TaskStatus UpdateParticleGIDs2(Driver *pdrive, int stage);
  TaskStatus CountParticleSends2(Driver *pdrive, int stage);
  TaskStatus InitParticleRecv2(Driver *pdrive, int stage);
  TaskStatus SendParticles2(Driver *pdrive, int stage);
  TaskStatus RecvParticles2(Driver *pdrive, int stage);
  TaskStatus RemoveEscaped(Driver *pdrive, int stage);   // after the first migration
  TaskStatus RemoveEscaped2(Driver *pdrive, int stage);  // after the second migration
  void RemoveEscapedNow();
  TaskStatus AddDragHistoryGas(Driver *pdrive, int stage); // u0 += a_twid*dt*R_g
  TaskStatus DepositDrag(Driver *pdrive, int stage);       // scatter Q,P
  TaskStatus SendDepQP(Driver *pdrive, int stage);
  TaskStatus RecvDepQP(Driver *pdrive, int stage);
  TaskStatus FoldDepQP(Driver *pdrive, int stage);
  TaskStatus GasImplicitSolve(Driver *pdrive, int stage);  // u* = (rho*u+P)/(rho+Q)
  void UpdateHybridMetric(Driver *pdrive, int stage);
  TaskStatus SolveCoupledStage(Driver *pdrive, int stage);
  void ApplyCoupledOperator(DvceArray5D<Real> &field, DvceArray5D<Real> &result,
                            Real a_dt);
  void ApplyDefectCorrection(DvceArray5D<Real> &field, DvceArray5D<Real> &work,
                             Real a_dt);
  // Internal shared implementation.  This must remain public because it encloses NVCC
  // extended host/device lambdas; callers should use one of the explicit methods above.
  void ApplyCoupledOperatorImpl(DvceArray5D<Real> &field, DvceArray5D<Real> &result,
                                Real a_dt, bool correct_field);
  void CompleteCopyExchange(DvceArray5D<Real> &field);
  void CompleteAddExchange(DvceArray5D<Real> &field);
  void GlobalDot(DvceArray5D<Real> &left, DvceArray5D<Real> &right, Real value[3]);
  void GlobalPreconditionedNorm(DvceArray5D<Real> &residual, Real value[3]);
  int StrictPCG(Real a_dt, bool residual_is_current);
  Real AdaptiveErrorBound(Real a_dt, Real residual_norm, Real &state_scale,
                          Real &acceptance_target);
  TaskStatus SendUstar(Driver *pdrive, int stage);
  TaskStatus RecvUstar(Driver *pdrive, int stage);
  TaskStatus SendUstarShr(Driver *pdrive, int stage);
  TaskStatus RecvUstarShr(Driver *pdrive, int stage);
  TaskStatus ProlongateUstar(Driver *pdrive, int stage);   // SMR: fine ghosts of u*
  TaskStatus GatherKickPMBR(Driver *pdrive, int stage);    // gather+kick+record+scatter
  TaskStatus SendPMBR(Driver *pdrive, int stage);
  TaskStatus RecvPMBR(Driver *pdrive, int stage);
  TaskStatus FoldPMBR(Driver *pdrive, int stage);
  Real ShearOffset() const;   // q*Omega*Lx*time: the shear-periodic y offset (a length)
  TaskStatus ApplyPMBR(Driver *pdrive, int stage);         // u0 += dmom; dmom -> R_g
  TaskStatus NewTimeStep(Driver *pdrive, int stage);
  TaskStatus NewTimeStep2(Driver *pdrive, int stage);
  TaskStatus ComputeNewTimeStep(Driver *pdrive, int stage);
  // ...self-gravity (Phase 4c, dust_gravity.cpp): "before_stagen" deposit + exchanges,
  // then the driver's Poisson solve, then in "stagen" the force field + exchanges
  bool GravityStage(Driver *pdrive, int stage) const;
  TaskStatus DepositGravity(Driver *pdrive, int stage);    // rho_dust = sum m W / V
  TaskStatus SendRhoDust(Driver *pdrive, int stage);
  TaskStatus RecvRhoDust(Driver *pdrive, int stage);
  TaskStatus FoldRhoDust(Driver *pdrive, int stage);
  TaskStatus SendRhoDustCopy(Driver *pdrive, int stage);
  TaskStatus RecvRhoDustCopy(Driver *pdrive, int stage);
  TaskStatus SendRhoDustShr(Driver *pdrive, int stage);
  TaskStatus RecvRhoDustShr(Driver *pdrive, int stage);
  TaskStatus ComputeGravForce(Driver *pdrive, int stage);  // gforce = -grad(phi)
  TaskStatus SendGravForce(Driver *pdrive, int stage);
  TaskStatus RecvGravForce(Driver *pdrive, int stage);
  TaskStatus SendGravForceShr(Driver *pdrive, int stage);
  TaskStatus RecvGravForceShr(Driver *pdrive, int stage);
  TaskStatus ProlongateRhoDust(Driver *pdrive, int stage);   // SMR: fine ghost layer
  TaskStatus ProlongateGravForce(Driver *pdrive, int stage); // SMR: fine ghosts of g
  // SMR helpers around a copy exchange of a ghost-filled field: restrict the active zone
  // into the coarse copy before PackAndSendCC; after the receives (and the shear remap)
  // fill the coarse array in same-level boundaries, apply zero-gradient values in the
  // coarse ghosts at physical faces, and prolongate the fine ghosts next to coarser
  // neighbors.  No-ops on a uniform mesh.
  void RestrictField(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  // SMR finest-level deposits (dust_smr.cpp): zero a fine image before a scatter, and
  // volume-average it onto the block cells (active zone + ghosts) after; both no-ops
  // when no block of this pack has rfac = 2
  void ZeroImage(DvceArray5D<Real> &fimg);
  void RestrictImage(DvceArray5D<Real> &fimg, DvceArray5D<Real> &a);
  void ProlongateField(MeshBoundaryValuesCC *pbval, DvceArray5D<Real> &a,
                       DvceArray5D<Real> &ca);
  void DepositMass();                // the deposit kernel (rho_dust, active + ghosts)
  void ComputeForceField();          // gforce on active cells from phi
  void AssembleDustDensityNow();   // synchronous deposit + exchanges (static solves)
  void ComputeGravForceNow();        // synchronous force field + exchanges
  // ...in "after_stagen" list
  TaskStatus ClearParticleSend(Driver *pdrive, int stage);
  TaskStatus ClearParticleRecv(Driver *pdrive, int stage);
  TaskStatus ClearDep(Driver *pdrive, int stage);

 private:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this DustGasDrag
};

} // namespace dust
#endif // DUST_DUST_HPP_
