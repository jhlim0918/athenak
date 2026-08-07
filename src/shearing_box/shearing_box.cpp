//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.cpp
//! \brief constructor for ShearingBox abstract base class, and utility functions

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "shearing_box.hpp"

//----------------------------------------------------------------------------------------
//! ShearingBox base class constructor

ShearingBox::ShearingBox(MeshBlockPack *ppack, ParameterInput *pin) :
    nmb_x1bndry("nmbx1",2),
    x1bndry_mbgid("x1gid",1,1),
    shearing_box_r_phi(false),     // 2D r-phi not yet implemented
    pmy_pack(ppack) {
  // Read shear rate and orbital frequency
  qshear = pin->GetReal("shearing_box","qshear");
  omega0 = pin->GetReal("shearing_box","omega0");
  is_stratified = pin->GetOrAddBoolean("shearing_box","stratified",false);
  orbital_advection = pin->GetOrAddBoolean("shearing_box","orbital_advection",true);

#if MPI_PARALLEL_ENABLED
  // request arrays are (re)allocated in SetX1BndryMBs
  for (int n=0; n<2; ++n) {
    sendbuf[n].vars_req = nullptr;
    recvbuf[n].vars_req = nullptr;
  }
  // create unique communicators for shearing box
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_sbox);
#endif

  // build lists of MBs touching the shear-periodic x1 boundaries
  SetX1BndryMBs();
}

//----------------------------------------------------------------------------------------
// ShearingBox base class destructor

ShearingBox::~ShearingBox() {
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<2; ++n) {
    delete [] sendbuf[n].vars_req;
    delete [] recvbuf[n].vars_req;
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::SetX1BndryMBs()
//! \brief Build lists with the GID of every MB on this rank at the ix1/ox1 shearing-box
//! boundaries, and (with MPI) allocate the request arrays sized to those lists.  Called
//! by the constructor and again after every AMR mesh update, since AMR renumbers GIDs
//! and re-assigns MBs to ranks (MeshBlockPack::pmb is rebuilt before this is called).

void ShearingBox::SetX1BndryMBs() {
  // Create vector with GID of every MBs on this rank at ix1/ox1 shearing-box boundaries
  std::vector<int> tmp_ix1bndry_gid, tmp_ox1bndry_gid;
  auto &mbbcs = pmy_pack->pmb->mb_bcs;
  for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
    if (mbbcs.h_view(m,BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      tmp_ix1bndry_gid.push_back(m + pmy_pack->gids);
    }
    if (mbbcs.h_view(m,BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      tmp_ox1bndry_gid.push_back(m + pmy_pack->gids);
    }
  }
  // number of MBs at ix1/ox1 boundaries is size of vectors
  nmb_x1bndry(0) = tmp_ix1bndry_gid.size();
  nmb_x1bndry(1) = tmp_ox1bndry_gid.size();

  // allocate mbgid array (grow-only across rebuilds) and initialize GIDs to -1
  // Ensure nmb is at least 1 to avoid zero-sized allocations
  int nmb = std::max(1, std::max(nmb_x1bndry(0),nmb_x1bndry(1)));
  if (static_cast<int>(x1bndry_mbgid.h_view.extent(1)) < nmb) {
    Kokkos::realloc(x1bndry_mbgid, 2, nmb);
  }
  int nmb_alloc = x1bndry_mbgid.h_view.extent(1);
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_alloc; ++m) {
      x1bndry_mbgid.h_view(n,m) = -1;
    }
  }
  // load GIDs of meshblocks at x1 boundaries into DualArray
  for (int m=0; m<nmb_x1bndry(0); ++m) {
    x1bndry_mbgid.h_view(0,m) = tmp_ix1bndry_gid[m];
  }
  for (int m=0; m<nmb_x1bndry(1); ++m) {
    x1bndry_mbgid.h_view(1,m) = tmp_ox1bndry_gid[m];
  }
  // sync with device
  x1bndry_mbgid.template modify<HostMemSpace>();
  x1bndry_mbgid.template sync<DevExeSpace>();

#if MPI_PARALLEL_ENABLED
  // (re)allocate vectors of MPI requests for ix1/ox1 boundaries in fixed length arrays
  // each MB on x1-face can communicate with up to 3 nghbrs.  All communications are
  // complete whenever this is called, so simply reset every request to MPI_REQUEST_NULL
  for (int n=0; n<2; ++n) {
    delete [] sendbuf[n].vars_req;
    delete [] recvbuf[n].vars_req;
    sendbuf[n].vars_req = nullptr;
    recvbuf[n].vars_req = nullptr;
    if (nmb_x1bndry(n) > 0) {
      sendbuf[n].vars_req = new MPI_Request[3*nmb_x1bndry(n)];
      recvbuf[n].vars_req = new MPI_Request[3*nmb_x1bndry(n)];
      for (int m=0; m<nmb_x1bndry(n); ++m) {
        for (int l=0; l<3; ++l) {
          sendbuf[n].vars_req[3*m + l] = MPI_REQUEST_NULL;
          recvbuf[n].vars_req[3*m + l] = MPI_REQUEST_NULL;
        }
      }
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::ReinitAfterMeshUpdate()
//! \brief Rebuild x1-boundary MB lists and resize communication buffers after an AMR
//! mesh update.  Must be called after MeshBlockPack::AddMeshBlocks/SetNeighbors have
//! rebuilt the MB metadata, and before any shear-periodic boundary exchange on the
//! new mesh.

void ShearingBox::ReinitAfterMeshUpdate() {
  SetX1BndryMBs();
  AllocateBuffers();
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::FindTargetMB()
//! \brief  function to find target MB offset by shear.  Returns GID and rank

void ShearingBox::FindTargetMB(const int igid, const int jshift, int &gid,
                                       int &rank) {
  Mesh *pm = pmy_pack->pmesh;
  // find lloc of input MB
  LogicalLocation lloc = pm->lloc_eachmb[igid];
  // find number of MBs in x2 direction at this level
  std::int32_t nmbx2 = pm->nmb_rootx2 << (lloc.level - pm->root_level);
  // apply shift by input number of blocks
  lloc.lx2 = static_cast<std::int32_t>((lloc.lx2 + jshift) % nmbx2);
  if (lloc.lx2 < 0) lloc.lx2 += nmbx2;
  // find target GID and rank
  gid = (pm->ptree->FindMeshBlock(lloc))->GetGID();
  rank = pm->rank_eachmb[gid];
  return;
}
