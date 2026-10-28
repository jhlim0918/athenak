#ifndef SHEARING_BOX_ORBITAL_ADVECTION_HPP_
#define SHEARING_BOX_ORBITAL_ADVECTION_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection.hpp
//! \brief definitions for classes that implement orbital advection abstract base and
//! derived classes (for CC and FC variables).

#include "athena.hpp"
#include "parameter_input.hpp"
#include "shearing_box/shearing_box.hpp"
#include "tasklist/task_list.hpp"

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvection
//  \brief Abstract base class for orbital advection of CC and FC variables

class OrbitalAdvection {
 public:
  OrbitalAdvection(MeshBlockPack *ppack, ParameterInput *pin);
  virtual ~OrbitalAdvection();

  // data
  int maxjshift;            // maximum integer shift of any cell in orbital advection
  Real qshear, omega0;      // Copies needed for all OA functions
  bool shearing_box_r_phi;  // NOT YET IMPLEMENTED

  // data buffers for orbital advection. Only two x2-faces communicate
  ShearingBoxBoundaryBuffer sendbuf[2], recvbuf[2];

#if MPI_PARALLEL_ENABLED
  // unique MPI communicator for orbital advection
  MPI_Comm comm_orb_advect;
#endif

  // functions (virtual: the CC class overrides them for its per-subface buffers)
  virtual TaskStatus InitRecv();
  virtual TaskStatus ClearRecv();
  virtual TaskStatus ClearSend();

 protected:
  // must use pointer to MBPack and not parent physics module since parent can be one of
  // many types (Hydro, MHD, Radiation, etc.)
  MeshBlockPack *pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvectionCC
//  \brief Derived class implementing orbital advection of CC variables

//
//  The x2-face neighbours of a MeshBlock may sit one level coarser or finer (arbitrary
//  refinement in x2). Each block remaps on its own grid, using an extended column filled
//  with the x2 neighbour's data brought to the block's resolution:
//    same level: the nrow rows next to the face are copied;
//    finer:      each of the 4 fine blocks on the face restricts its rows and sends its
//                (x1,x3) quadrant;
//    coarser:    the coarse block sends a box of its rows (with one ghost cell on every
//                side, for slopes) and the fine block prolongates it (as ProlongCC).
//  Buffers are one flat array per (MeshBlock, face): a message for sub-face q starts at
//  q*(message size). After the remap, fine blocks send their remap "fluxes" through the
//  shared face (restricted) to the coarse neighbour, which replaces its own; this keeps
//  the remap exactly conservative across level jumps (OA flux correction).

class OrbitalAdvectionCC : public OrbitalAdvection {
 public:
  OrbitalAdvectionCC(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
  ~OrbitalAdvectionCC();

  int nvar_;            // number of variables communicated
  int nrow;             // rows (receiver cells) communicated across a same-level x2 face
  int nrow_fc;          // coarse rows a fine block sends to a coarser x2 neighbour
  int nrow_cf;          // coarse rows a coarse block sends to a finer x2 neighbour
  int sz_same, sz_fc, sz_cf, sz_flx;  // message sizes per variable
  int sz_face;          // size of one (MeshBlock, face) buffer, all variables
  bool flux_correct;    // OA flux correction at x2 level jumps (<shearing_box>
                        // oa_flux_correct, default true; false only for tests)

  // buffers: (nmb, 2 faces, sz_face) for data; (nmb, 2, 4*nvar*sz_flx) for fluxes
  DvceArray3D<Real> sbuf_oa, rbuf_oa, sflx_oa, rflx_oa;
  // remap fluxes through this block's own x2 faces: (nmb, 2, nvar, nx3, nx1)
  DvceArray5D<Real> gflx;

#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_orb_flx;
  int nreq;                                 // requests per array = nmb*8 sub-faces
  MPI_Request *sreq, *rreq, *sreq_flx, *rreq_flx;
#endif

  // functions to communicate CC data with orbital advection
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, ReconstructionMethod rcon);
  TaskStatus SendFluxCC();
  TaskStatus RecvAndCorrectFluxCC(DvceArray5D<Real> &a);
  int RowsAvailable(int m);   // rows the x2 neighbours of MeshBlock m can supply
  Real MaxTimeStep();         // largest dt whose shift stays within those rows
  TaskStatus InitRecv() override;
  TaskStatus ClearRecv() override;
  TaskStatus ClearSend() override;
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvectionFC
//  \brief Derived class implementing orbital advection of FC variables

//
//  With mesh refinement the remap's effective EMFs are corrected like the stage EMFs
//  before the field is updated: they are converted to E*dt on the edges (eoa) and passed
//  through a private MeshBoundaryValuesFC flux correction, so a coarse block uses the
//  restricted EMFs of its finer neighbours on every edge they share, and both copies of
//  a shared face field stay equal. (x2-face neighbours must still be at the same level:
//  <shearing_box> oa_annular = true.)

class MeshBoundaryValuesFC;

class OrbitalAdvectionFC : public OrbitalAdvection {
 public:
  OrbitalAdvectionFC(MeshBlockPack *ppack, ParameterInput *pin);
  ~OrbitalAdvectionFC();

  int nrowb;            // rows of B1, B3 sent across an x2 face (at most nx2)
  bool flux_correct;    // EMF correction of the remap at level jumps (multilevel only)

  // containers for Ex and Ez used to remap magnetic fields (in units of dx2)
  DvceArray4D<Real> emfx, emfz;
  // the same as E*dt on the edges, for the EMF correction (multilevel only)
  DvceEdgeFld4D<Real> eoa;
  MeshBoundaryValuesFC *pbval_oa = nullptr;

  // functions to communicate FC data with orbital advection
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b0, ReconstructionMethod rcon);
  TaskStatus SendEMF_OA();
  TaskStatus RecvEMFAndUpdateB_OA(DvceFaceFld4D<Real> &b0);
  TaskStatus InitRecv() override;
  TaskStatus ClearRecv() override;
  TaskStatus ClearSend() override;

 private:
  void UpdateB(DvceFaceFld4D<Real> &b0);   // CT update of B from emfx, emfz
};

#endif // SHEARING_BOX_ORBITAL_ADVECTION_HPP_
