//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection_tasks.cpp
//! \brief functions included in task lists to post/clear non-blocking MPI calls for
//! orbital advection

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "shearing_box.hpp"
#include "orbital_advection.hpp"
#include "bvals/bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications with
//! orbital advection

TaskStatus OrbitalAdvection::InitRecv() {
#if MPI_PARALLEL_ENABLED
  const int &nmb = pmy_pack->nmb_thispack;
  const auto &nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications of variables
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if (nghbr.h_view(m,nnghbr).gid >= 0) {
        // rank of neighboring MeshBlock sending data
        int srank = nghbr.h_view(m,nnghbr).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (srank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, nnghbr);

          // get pointer to variables
          using Kokkos::ALL;
          auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, ALL, ALL, ALL, ALL);
          int data_size = recv_ptr.size();

          // Post non-blocking receive for this buffer on this MeshBlock
          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, srank, tag,
                               comm_orb_advect, &(recvbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void OrbitalAdvection::ClearRecv
//! \brief Waits for all MPI receives associated with communcation with orbital
//! advection to complete before allowing execution to continue

TaskStatus OrbitalAdvection::ClearRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  int &nmb = pmy_pack->nmb_thispack;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for vars to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if ( (nghbr.h_view(m,nnghbr).gid >= 0) &&
           (nghbr.h_view(m,nnghbr).rank != global_variable::my_rank) ) {
        int ierr = MPI_Wait(&(recvbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void OrbitalAdvection::ClearSend
//! \brief Waits for all MPI sends associated with communcation of boundary variables
//! to complete before allowing execution to continue

TaskStatus OrbitalAdvection::ClearSend() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  int &nmb = pmy_pack->nmb_thispack;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for vars to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if ( (nghbr.h_view(m,nnghbr).gid >= 0) &&
           (nghbr.h_view(m,nnghbr).rank != global_variable::my_rank) ) {
        int ierr = MPI_Wait(&(sendbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::InitRecv
//! \brief Posts non-blocking receives (with MPI) for the CC orbital advection data from
//! every x2 sub-face neighbour, and (with refinement) for the remap fluxes from finer
//! x2 neighbours. Message sizes depend on the level of the neighbour.

TaskStatus OrbitalAdvectionCC::InitRecv() {
#if MPI_PARALLEL_ENABLED
  const int &nmb = pmy_pack->nmb_thispack;
  const auto &nghbr = pmy_pack->pmb->nghbr;
  const auto &mblev = pmy_pack->pmb->mb_lev;
  const int nvar = nvar_;

  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int s=0; s<8; ++s) {
      int nn = 8 + s;
      if (nghbr.h_view(m,nn).gid >= 0) {
        int srank = nghbr.h_view(m,nn).rank;
        if (srank != global_variable::my_rank) {
          int f = s/4, q = s - 4*f;
          int nlev = nghbr.h_view(m,nn).lev, mylev = mblev.h_view(m);
          // receiver's view: a finer neighbour sends restricted rows (sz_fc), a coarser
          // one a coarse box (sz_cf)
          int sz = (nlev == mylev) ? sz_same : ((nlev > mylev) ? sz_fc : sz_cf);
          int tag = CreateBvals_MPI_Tag(m, nn);
          auto recv_ptr = Kokkos::subview(rbuf_oa, m, f,
                              std::make_pair(q*nvar*sz, (q+1)*nvar*sz));
          int ierr = MPI_Irecv(recv_ptr.data(), nvar*sz, MPI_ATHENA_REAL, srank, tag,
                               comm_orb_advect, &(rreq[8*m+s]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (flux_correct && (nlev > mylev)) {
            auto rflx_ptr = Kokkos::subview(rflx_oa, m, f,
                                std::make_pair(q*nvar*sz_flx, (q+1)*nvar*sz_flx));
            ierr = MPI_Irecv(rflx_ptr.data(), nvar*sz_flx, MPI_ATHENA_REAL, srank, tag,
                             comm_orb_flx, &(rreq_flx[8*m+s]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void OrbitalAdvectionCC::ClearRecv
//! \brief Waits for all MPI receives of CC orbital advection data and fluxes

TaskStatus OrbitalAdvectionCC::ClearRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  int nmb = pmy_pack->nmb_thispack;
  for (int n=0; n<8*nmb; ++n) {
    if (MPI_Wait(&(rreq[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) {no_errors=false;}
    if (MPI_Wait(&(rreq_flx[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) {no_errors=false;}
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void OrbitalAdvectionCC::ClearSend
//! \brief Waits for all MPI sends of CC orbital advection data and fluxes

TaskStatus OrbitalAdvectionCC::ClearSend() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  int nmb = pmy_pack->nmb_thispack;
  for (int n=0; n<8*nmb; ++n) {
    if (MPI_Wait(&(sreq[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) {no_errors=false;}
    if (MPI_Wait(&(sreq_flx[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) {no_errors=false;}
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::InitRecv
//! \brief Posts receives for the B3/B1 rows of every x2 sub-face neighbour (sizes by
//! level) and, with mesh refinement, for the remap EMFs (private MeshBoundaryValuesFC)

TaskStatus OrbitalAdvectionFC::InitRecv() {
#if MPI_PARALLEL_ENABLED
  const int &nmb = pmy_pack->nmb_thispack;
  const auto &nghbr = pmy_pack->pmb->nghbr;
  const auto &mblev = pmy_pack->pmb->mb_lev;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int s=0; s<8; ++s) {
      int nn = 8 + s;
      if (nghbr.h_view(m,nn).gid >= 0) {
        int srank = nghbr.h_view(m,nn).rank;
        if (srank != global_variable::my_rank) {
          int f = s/4, q = s - 4*f;
          int nlev = nghbr.h_view(m,nn).lev, mylev = mblev.h_view(m);
          int sz = (nlev == mylev) ? szb_same : ((nlev > mylev) ? szb_fc : szb_cf);
          int tag = CreateBvals_MPI_Tag(m, nn);
          auto recv_ptr = Kokkos::subview(rbufb, m, f,
                              std::make_pair(q*2*sz, (q+1)*2*sz));
          int ierr = MPI_Irecv(recv_ptr.data(), 2*sz, MPI_ATHENA_REAL, srank, tag,
                               comm_orb_advect, &(rreqb[8*m+s]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  TaskStatus tstat = TaskStatus::complete;
  if (flux_correct) {tstat = pbval_oa->InitFluxRecv(3);}
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::ClearRecv

TaskStatus OrbitalAdvectionFC::ClearRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  int nmb = pmy_pack->nmb_thispack;
  for (int n=0; n<8*nmb; ++n) {
    if (MPI_Wait(&(rreqb[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) {no_errors=false;}
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  TaskStatus tstat = TaskStatus::complete;
  if (flux_correct) {tstat = pbval_oa->ClearFluxRecv();}
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::ClearSend

TaskStatus OrbitalAdvectionFC::ClearSend() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  int nmb = pmy_pack->nmb_thispack;
  for (int n=0; n<8*nmb; ++n) {
    if (MPI_Wait(&(sreqb[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) {no_errors=false;}
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  TaskStatus tstat = TaskStatus::complete;
  if (flux_correct) {tstat = pbval_oa->ClearFluxSend();}
  return tstat;
}
