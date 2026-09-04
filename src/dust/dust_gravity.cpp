//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_gravity.cpp
//! \brief Dust self-gravity (dust track, Phase 4c).  Once per active stage, BEFORE the
//! driver's Poisson solve ("before_stagen"), the particle masses are deposited with the
//! module's PM kernel into rho_dust = sum_p m_p W(x_p)/V, the ghost deposits are added
//! into the neighbours (with the shear-periodic fold), and the ghost layer of the
//! resulting density is filled by a copy exchange (with the shear remap) because the
//! multigrid solver loads the active zone plus one ghost layer of its source.  Gravity
//! adds rho_dust to the gas density through Gravity::RegisterExtraDensity, so the gas
//! feels the total potential through its usual source term and the two solvers need no
//! dust-specific code.  AFTER the solve ("stagen") the force -grad(phi) is formed on the
//! active cells by centred differences (phi is valid on the active zone plus one ghost
//! layer), exchanged like the provisional gas velocity u*, and gathered onto the
//! particles with the same PM kernel in ExplicitPush.  The dormant imex2+ assembly
//! stage deposits nothing and the driver skips the solve there (Driver::Execute).
//!
//! Ghost deposits that land outside a non-periodic (physical) mesh face have no
//! receiver and are discarded, and the density and force ghosts at such faces are the
//! adjacent active values (force) or zero (density) -- the policy of the Athena-C
//! reference; see the implementation document.

#include <iostream>
#include <cstdlib>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box/shearing_box.hpp"
#include "gravity/gravity.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

namespace {
void RequireDone(TaskStatus status, const char *operation) {
  if (status == TaskStatus::fail) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Dust gravity exchange failed during " << operation
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::GravityStage
//! \brief true on stages where the dust gravity coupling does work

bool DustGasDrag::GravityStage(Driver *pdrive, int stage) const {
  return gravity && ActiveStage(pdrive, stage);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::DepositMass
//! \brief rho_dust = sum_p m_p W(x_p)/V on the active zone plus the deposit halo.  The
//! particles sit inside their owning MeshBlocks (last stage's migration), so the TSC
//! stencil reaches at most one ghost layer.

void DustGasDrag::DepositMass() {
  Kokkos::deep_copy(DevExeSpace(), rho_dust, 0.0);

  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) {return;}

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto gids = pmy_pack->gids;
  int scheme = static_cast<int>(deposit);
  auto &rho = rho_dust;

  par_for("dust_grav_deposit",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int ip, jp, kp;
    Real wx[3], wy[3], wz[3];
    PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
              scheme, ip, wx);
    PMWeights(pr(IPY,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
              scheme, jp, wy);
    if (three_d) {
      PMWeights(pr(IPZ,p), mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, nx3, ks,
                scheme, kp, wz);
    } else {
      kp = ks;
      wz[0] = 0.0; wz[1] = 1.0; wz[2] = 0.0;
    }
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) {vol *= mbsize.d_view(m).dx3;}
    Real minv = pr(IPM,p)/vol;
    int clo = three_d ? 0 : 1, chi = three_d ? 2 : 1;
    for (int c=clo; c<=chi; ++c) {
      for (int b=0; b<3; ++b) {
        Real wcb = wz[c]*wy[b]*minv;
        if (wcb == 0.0) continue;
        for (int a=0; a<3; ++a) {
          DepositAdd(&rho(m,0,kp+c-1,jp+b-1,ip+a-1), wcb*wx[a]);
        }
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ComputeForceField
//! \brief gforce = -grad(phi) by centred differences on the active zone; at physical
//! (non-periodic, non-block) faces the ghost layers copy the adjacent active value so
//! that a particle within one cell of such a face still gathers a full stencil.

void DustGasDrag::ComputeForceField() {
  auto &phi = pmy_pack->pgrav->phi;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ng = indcs.ng;
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &g = gforce;

  par_for("dust_grav_force",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    g(m,0,k,j,i) = -(phi(m,0,k,j,i+1) - phi(m,0,k,j,i-1))/(2.0*mbsize.d_view(m).dx1);
    g(m,1,k,j,i) = multi_d ?
        -(phi(m,0,k,j+1,i) - phi(m,0,k,j-1,i))/(2.0*mbsize.d_view(m).dx2) : 0.0;
    g(m,2,k,j,i) = three_d ?
        -(phi(m,0,k+1,j,i) - phi(m,0,k-1,j,i))/(2.0*mbsize.d_view(m).dx3) : 0.0;
  });

  // physical faces: zero-gradient fill of the ghost layers (periodic, shear-periodic
  // and internal faces are filled by the exchanges that follow)
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  int n1 = g.extent_int(4), n2 = g.extent_int(3), n3 = g.extent_int(2);
  auto physical = [] (BoundaryFlag f) {
    return (f != BoundaryFlag::block) && (f != BoundaryFlag::periodic) &&
           (f != BoundaryFlag::shear_periodic);
  };
  par_for("dust_grav_force_bc",DevExeSpace(),0,nmb1,0,2,0,n3-1,0,n2-1,0,n1-1,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    int kk = k, jj = j, ii = i;
    bool in_ghost = false;
    auto &bc = mb_bcs.d_view;
    if (i < is && physical(bc(m,BoundaryFace::inner_x1))) {ii = is; in_ghost = true;}
    if (i > ie && physical(bc(m,BoundaryFace::outer_x1))) {ii = ie; in_ghost = true;}
    if (multi_d) {
      if (j < js && physical(bc(m,BoundaryFace::inner_x2))) {jj = js; in_ghost = true;}
      if (j > je && physical(bc(m,BoundaryFace::outer_x2))) {jj = je; in_ghost = true;}
    }
    if (three_d) {
      if (k < ks && physical(bc(m,BoundaryFace::inner_x3))) {kk = ks; in_ghost = true;}
      if (k > ke && physical(bc(m,BoundaryFace::outer_x3))) {kk = ke; in_ghost = true;}
    }
    if (in_ghost) {
      // clamp the remaining indices into the active zone (corner ghosts of a physical
      // face next to an internal face are overwritten by the exchange anyway)
      ii = (ii < is) ? is : ((ii > ie) ? ie : ii);
      jj = (jj < js) ? js : ((jj > je) ? je : jj);
      kk = (kk < ks) ? ks : ((kk > ke) ? ke : kk);
      g(m,v,k,j,i) = g(m,v,kk,jj,ii);
    }
  });
  (void)ng;
}

//----------------------------------------------------------------------------------------
// task wrappers: "before_stagen" density assembly

TaskStatus DustGasDrag::DepositGravity(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_source) {return TaskStatus::complete;}
  DepositMass();
  return TaskStatus::complete;
}

TaskStatus DustGasDrag::SendRhoDust(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_source) {return TaskStatus::complete;}
  return pbval_rd->PackAndSendDeposit(rho_dust);
}

TaskStatus DustGasDrag::RecvRhoDust(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_source) {return TaskStatus::complete;}
  return pbval_rd->RecvAndSumDeposit(rho_dust);
}

TaskStatus DustGasDrag::FoldRhoDust(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_source) {return TaskStatus::complete;}
  if (!shear_fold) {return TaskStatus::complete;}
  return pbval_rd->FoldShearDeposit(rho_dust, ShearOffset(), shear_remap);
}

TaskStatus DustGasDrag::SendRhoDustCopy(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_source) {return TaskStatus::complete;}
  return pbval_rc->PackAndSendCC(rho_dust, cdummy);
}

TaskStatus DustGasDrag::RecvRhoDustCopy(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_source) {return TaskStatus::complete;}
  return pbval_rc->RecvAndUnpackCC(rho_dust, cdummy);
}

TaskStatus DustGasDrag::SendRhoDustShr(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_source || psbox_rc == nullptr) {
    return TaskStatus::complete;
  }
  return psbox_rc->PackAndSendCC(rho_dust, shear_remap);
}

TaskStatus DustGasDrag::RecvRhoDustShr(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_source || psbox_rc == nullptr) {
    return TaskStatus::complete;
  }
  return psbox_rc->RecvAndUnpackCC(rho_dust);
}

//----------------------------------------------------------------------------------------
// task wrappers: "stagen" force field (after the driver's Poisson solve)

TaskStatus DustGasDrag::ComputeGravForce(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_force) {return TaskStatus::complete;}
  ComputeForceField();
  return TaskStatus::complete;
}

TaskStatus DustGasDrag::SendGravForce(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_force) {return TaskStatus::complete;}
  return pbval_g->PackAndSendCC(gforce, cdummy);
}

TaskStatus DustGasDrag::RecvGravForce(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_force) {return TaskStatus::complete;}
  return pbval_g->RecvAndUnpackCC(gforce, cdummy);
}

TaskStatus DustGasDrag::SendGravForceShr(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_force || psbox_g == nullptr) {
    return TaskStatus::complete;
  }
  return psbox_g->PackAndSendCC(gforce, ReconstructionMethod::plm);
}

TaskStatus DustGasDrag::RecvGravForceShr(Driver *pdrive, int stage) {
  if (!GravityStage(pdrive, stage) || !gravity_force || psbox_g == nullptr) {
    return TaskStatus::complete;
  }
  return psbox_g->RecvAndUnpackCC(gforce);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::AssembleDustDensityNow / ComputeGravForceNow
//! \brief Synchronous versions of the two task chains for problem generators that call
//! the Poisson solve outside the time loop (static solves, initial diagnostics).

void DustGasDrag::AssembleDustDensityNow() {
  Real time = pmy_pack->pmesh->time;
  DepositMass();
  RequireDone(pbval_rd->InitRecv(1), "density InitRecv");
  RequireDone(pbval_rd->PackAndSendDeposit(rho_dust), "density send");
  TaskStatus status;
  do {
    status = pbval_rd->RecvAndSumDeposit(rho_dust);
    RequireDone(status, "density receive");
  } while (status == TaskStatus::incomplete);
  if (shear_fold) {
    RequireDone(pbval_rd->FoldShearDeposit(rho_dust, ShearOffset(), shear_remap),
                "density fold");
  }
  RequireDone(pbval_rd->ClearSend(), "density ClearSend");
  RequireDone(pbval_rd->ClearRecv(), "density ClearRecv");

  RequireDone(pbval_rc->InitRecv(1), "density-copy InitRecv");
  RequireDone(pbval_rc->PackAndSendCC(rho_dust, cdummy), "density-copy send");
  do {
    status = pbval_rc->RecvAndUnpackCC(rho_dust, cdummy);
    RequireDone(status, "density-copy receive");
  } while (status == TaskStatus::incomplete);
  RequireDone(pbval_rc->ClearSend(), "density-copy ClearSend");
  RequireDone(pbval_rc->ClearRecv(), "density-copy ClearRecv");

  if (psbox_rc != nullptr) {
    RequireDone(psbox_rc->InitRecv(time), "density-shear InitRecv");
    RequireDone(psbox_rc->PackAndSendCC(rho_dust, shear_remap), "density-shear send");
    do {
      status = psbox_rc->RecvAndUnpackCC(rho_dust);
      RequireDone(status, "density-shear receive");
    } while (status == TaskStatus::incomplete);
    RequireDone(psbox_rc->ClearSend(), "density-shear ClearSend");
    RequireDone(psbox_rc->ClearRecv(), "density-shear ClearRecv");
  }
}

void DustGasDrag::ComputeGravForceNow() {
  if (!gravity || !gravity_force) {return;}
  Real time = pmy_pack->pmesh->time;
  ComputeForceField();
  RequireDone(pbval_g->InitRecv(3), "force InitRecv");
  RequireDone(pbval_g->PackAndSendCC(gforce, cdummy), "force send");
  TaskStatus status;
  do {
    status = pbval_g->RecvAndUnpackCC(gforce, cdummy);
    RequireDone(status, "force receive");
  } while (status == TaskStatus::incomplete);
  RequireDone(pbval_g->ClearSend(), "force ClearSend");
  RequireDone(pbval_g->ClearRecv(), "force ClearRecv");
  if (psbox_g != nullptr) {
    RequireDone(psbox_g->InitRecv(time), "force-shear InitRecv");
    RequireDone(psbox_g->PackAndSendCC(gforce, ReconstructionMethod::plm),
                "force-shear send");
    do {
      status = psbox_g->RecvAndUnpackCC(gforce);
      RequireDone(status, "force-shear receive");
    } while (status == TaskStatus::incomplete);
    RequireDone(psbox_g->ClearSend(), "force-shear ClearSend");
    RequireDone(psbox_g->ClearRecv(), "force-shear ClearRecv");
  }
}

} // namespace dust
