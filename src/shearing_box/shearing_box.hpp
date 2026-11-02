#ifndef SHEARING_BOX_SHEARING_BOX_HPP_
#define SHEARING_BOX_SHEARING_BOX_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shearing_box.hpp
//! \brief definitions for classes that implement shearing box abstract base and derived
//! classes (for CC and FC variables).

#include <array>
#include <cstdint>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

//----------------------------------------------------------------------------------------
//! \struct ShearingBoxTaskIDs
//  \brief container to hold TaskIDs of all shearing_box tasks

struct ShearingBoxTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID csend;
  TaskID crecv;
};

//----------------------------------------------------------------------------------------
//! \struct ShearingBoxBoundaryBuffer
//! \brief container for storing boundary buffers. Used by both the orbital advection and
//! shearing box methods for both CC and FC variables.
//! Basically a much simplified version of the MeshBoundaryBuffer struct.

struct ShearingBoxBoundaryBuffer {
  // Views that store buffer data and fluxes on device
  DvceArray5D<Real> vars, flux;
#if MPI_PARALLEL_ENABLED
  // vectors of length (number of MBs) to hold MPI requests
  // Using STL vector causes problems with some GPU compilers, so just use plain C array
  MPI_Request *vars_req, *flux_req;
#endif
};

//----------------------------------------------------------------------------------------
//! \class ShearingBox
//  \brief Abstract base class for shearing box boundaries with CC and FC vars

class ShearingBox {
 public:
  ShearingBox(MeshBlockPack *ppack, ParameterInput *pin);
  ~ShearingBox();

  // data
  HostArray1D<int> nmb_x1bndry;    // number of MBs that touch x1 boundaries
  DualArray2D<int> x1bndry_mbgid;  // GIDs of MBs at x1 boundaries
  Real yshear;                     // x2-distance x1-boundaries have sheared
  Real qshear, omega0;             // Copies needed for all SB funcs
  bool shearing_box_r_phi;         // NOT YET IMPLEMENTED
  bool is_stratified;              // true for stratified shearing box
  bool orbital_advection;          // false = non-FARGO mode: velocities include the
                                   // background shear, full-velocity source terms, and
                                   // shear-periodic BCs offset azimuthal momentum

  // data buffers for shearing box BCs.  Only two x1-faces get sheared
  // Use separate variables for ix1/ox1 since number of MBs on each face can be different
  ShearingBoxBoundaryBuffer sendbuf[2], recvbuf[2];

#if MPI_PARALLEL_ENABLED
  // unique MPI communicator for shearing box
  MPI_Comm comm_sbox;
#endif

  // ---- x1 faces whose MeshBlocks change level along x2 (port of R. Li's amrSG_dev,
  // shearing_box_pull.cpp / shearing_box_plan.cpp).  Set only by Hydro, and only when
  // Mesh::sbox_face_levels is true (hydro without MHD, orbital advection on, mesh
  // refinement in 3-D, no gravity/particles; see Mesh::Mesh).  Otherwise (MHD, uniform
  // meshes, ...) the stock same-level two-step exchange below is used unchanged.
  // face_levels: every block on a shear-periodic x1 face collects its x1 ghost zones
  // (and, above the root level, those of its coarse array) from the opposite face,
  // converted to its level by the senders (averaged if finer, limited-slope prolongation
  // if coarser) and remapped by the receiver (PlanPull, ShearingBoxCC::PullSend/PullRecv).
  bool face_levels = false;
  bool coarse_on = false;          // face_levels: also the coarse arrays' x1 ghosts
  bool x1_plans = false;           // block-local candidate lists needed (pull or flux)
  bool pull_ghost_valid = false;   // x1 ghosts hold collected values (not after a regrid)
  int nvar_pull = 0;
  ReconstructionMethod rcon_pull = ReconstructionMethod::plm;
  enum PullField {kPRGid, kPRType, kPRFace, kPRIdx, kPSGid, kPSLoc, kPSFace, kPD,
                  kPY0, kPNy, kPYS, kPZ0, kPNz, kPZS, kPOff, kPSeq, kPNField};
  DualArray2D<int> ppiece;      // pieces with a local sender or receiver
  int nppiece = 0, npymax = 1, npzmax = 1;
  DualArray2D<int> precv;       // local receivers: (m, type 0 fine / 1 coarse, face)
  DualArray1D<Real> precv_eps;  // fraction of the shift of each receiver
  int nprecv = 0;
  DvceArray5D<Real> pbuf;       // (receiver, v, z rows, depth, y rows)
  DvceArray1D<Real> psendflat, precvflat;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_pull;
  MPI_Request *psreq_ = nullptr, *prreq_ = nullptr;
  int npreq_ = 0;
  std::vector<int> pmsg_rank, pmsg_off, pmsg_size, pmsg_tag, pmsg_send;  // one per message
#endif
  void PlanPull();              // with the current yshear; posts the receives
  bool WaitPull();
  // Per regrid, every local block on an x1 face keeps the blocks of the opposite face
  // that overlap it in x3 (with a ghost margin): the only candidates of its shear
  // exchanges, whatever the shift.
  struct ShearCand {int gid, rank, level; std::int32_t lx2, lx3;};
  std::vector<std::vector<ShearCand>> cand_[2];   // [face of the local block][local m]
  std::vector<std::vector<int>> candsort_[2];      // their indices sorted (level,lx3,lx2)
  void BuildCandidates();
  // remote pieces of one exchange: entries {dir (0 receive, 1 send), peer, k1..k5, size};
  // fills offset[i] per piece and the message lists (rank, offset, size, send)
  static void LayoutPeerMessages(const std::vector<std::array<long long, 8>> &key,
                                 std::vector<int> &offset, std::vector<int> &mrank,
                                 std::vector<int> &moff, std::vector<int> &msize,
                                 std::vector<int> &msend, int &soff, int &roff);

  // functions
  TaskStatus InitRecv(Real time);
  TaskStatus ClearRecv();
  TaskStatus ClearSend();
  // function to find target MB offset by shear.  Returns GID and rank
  void FindTargetMB(const int igid, const int jshift, int &gid, int &rank);
  // globally consistent x2 cell width for integer shift arithmetic (per-block
  // dx2 roundoff can desynchronize sender/receiver message sizes across ranks)
  Real ConsistentDx2(const int gid) const;
  // function to find index in x1bndry array of MB with input GID
  int TargetIndex(const int n, const int tgid) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      if (x1bndry_mbgid.h_view(n,m) == tgid) return m;
    }
    return -1;
  }
  // rebuild x1-boundary MB lists and communication buffers after the mesh changes
  // (AMR renumbers GIDs and re-assigns MBs to ranks on every update)
  void ReinitAfterMeshUpdate();

 protected:
  // build list of GIDs of MBs touching the shear-periodic x1 boundaries (and MPI
  // request arrays sized to it) from the current MeshBlockPack; called by the
  // constructor and again after every AMR mesh update
  void SetX1BndryMBs();
  // (re)allocate send/recv buffers sized to nmb_x1bndry; implemented by CC/FC
  // derived classes since buffer shapes differ
  virtual void AllocateBuffers() = 0;

  // must use pointer to MBPack and not parent physics module since parent can be one of
  // many types (Hydro, MHD, Radiation, etc.)
  MeshBlockPack *pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class ShearingBoxCC
//! \brief Derived class implementing shearing box boundary conditions and source terms
//! for CC vars

class ShearingBoxCC : public ShearingBox {
 public:
  ShearingBoxCC(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
  int nvar;   // number of CC variables communicated (needed to size buffers)
  // functions to communicate CC data with shearing box BCs
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a, ReconstructionMethod rcon);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a);
  void AllocateBuffers() override;
  // the coarse arrays' x1 ghost zones (face_levels only; collected by PullSend)
  TaskStatus RecvAndUnpackCoarseCC(DvceArray5D<Real> &ca);

  // ---- matching of the x1 fluxes across the shear-periodic boundary (port of R. Li's
  // shearing_box_flux.cpp): on each face column the finer side is authoritative (equal
  // levels: the outer face); every block on the other side replaces its x1 face fluxes
  // by the authoritative ones, restricted to its level if finer and remapped by the
  // shear offset, so mass and momentum leave one face exactly as they enter the other.
  // Set only by Hydro (hydro with orbital advection on a refined 3-D mesh).
  bool flux_on = false;
  bool flux_planned = false;    // PlanFluxMatch ran for the current stage
  enum FluxPieceField {kRGid, kSGid, kSFace, kR0, kLen, kDst, kKofs, kNk, kRestr, kQ,
                       kOff, kRIdx, kSLoc, kRRank, kSRank, kNField};
  DualArray2D<int> fpiece;      // pieces with a local sender or a local receiver
  int nfpiece = 0;
  DualArray2D<int> frecv_list;  // local receivers: (m, face); shift fraction in frecv_eps
  DualArray1D<Real> frecv_eps;
  int nfrecv = 0;
  DvceArray4D<Real> fbuf;       // receiver strips (m, v, k, nx2 + 6 rows)
  DvceArray1D<Real> fsendflat, frecvflat;
#if MPI_PARALLEL_ENABLED
  MPI_Request *fsreq_ = nullptr, *frreq_ = nullptr;
  int nfreq_ = 0;
  std::vector<int> fmsg_rank, fmsg_off, fmsg_size, fmsg_send, fmsg_tag;   // per message
#endif
  void PlanFluxMatch();                     // with the current yshear (InitRecv)
  TaskStatus SendFluxMatch(DvceFaceFld5D<Real> &flx);
  TaskStatus RecvFluxMatch(DvceFaceFld5D<Real> &flx, ReconstructionMethod rcon);
  TaskStatus ClearFluxMatch();
  // face levels: flux matching by overlapping cell pairs (finer side, equal: outer);
  // a block may receive on parts of its face and send on others
  enum FluxPairField {kQRGid, kQRIdx, kQRFace, kQSGid, kQSLoc, kQSFace, kQD, kQR0, kQNy,
                      kQK0, kQNz, kQBase, kQBaseZ, kQOff, kQPair, kQNField};
  DualArray2D<int> qpiece;
  DualArray1D<Real> qfrac;      // fractional part of the shift of each piece
  int nqpiece = 0;
  // (m, face, nvar+2, k, j): sum of contributions, then the covered fraction and its
  // first moment in x2; qnew the replaced fluxes before they are copied back
  DvceArray5D<Real> qacc, qnew;
  DvceArray1D<Real> qsendflat, qrecvflat;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_qflx;
  MPI_Request *qsreq_ = nullptr, *qrreq_ = nullptr;
  int nqreq_ = 0;
  std::vector<int> qmsg_rank, qmsg_off, qmsg_size, qmsg_tag, qmsg_send;
#endif
  void PlanFluxPair();
  TaskStatus SendFluxPair(DvceFaceFld5D<Real> &flx);
  TaskStatus RecvFluxPair(DvceFaceFld5D<Real> &flx);
  // the collected x1 ghost zones (face_levels)
  TaskStatus PullSend(DvceArray5D<Real> &a);
  TaskStatus PullRecv(DvceArray5D<Real> &a, const bool coarse);
  ~ShearingBoxCC();

  // shearing box source terms for Hydro CC variables
  void SourceTermsCC(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                     const Real bdt, DvceArray5D<Real> &u0);
  // shearing box source terms for MHD CC variables
  void SourceTermsCC(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                     const EOS_Data &eos_data, const Real bdt, DvceArray5D<Real> &u0);
};

//----------------------------------------------------------------------------------------
//! \class ShearingBoxFC
//!  \brief Derived class implementing shearing box boundary conditions and source terms
//! (if needed) for FC vars

class ShearingBoxFC : public ShearingBox {
 public:
  ShearingBoxFC(MeshBlockPack *ppack, ParameterInput *pin);
  // functions to communicate CC data with shearing box BCs
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b, ReconstructionMethod rcon);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b);
  void AllocateBuffers() override;
  // shearing box source terms for FC variables
  void SourceTermsFC(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
};

#endif // SHEARING_BOX_SHEARING_BOX_HPP_
