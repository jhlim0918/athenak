//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection_cc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! variables in the orbital advection step. Data is shifted by the appropriate offset
//! during the recv/unpack step, so these functions both communicate the data and perform
//! the shift.
//!
//! The x2-face neighbours may be one level coarser or finer than the MeshBlock (see the
//! class comment in orbital_advection.hpp). Buffer layout, per (MeshBlock, face f):
//!   f = 0 is the inner x2 face (nghbr slots 8-11), f = 1 the outer (slots 12-15);
//!   the message for sub-face q (slot 8+4f+q) starts at q*nvar*(message size), and holds
//!   variable v at v*(message size) + (k, row, i) with i fastest and rows in ascending j.
//! Message kinds (sender -> receiver):
//!   same level: nx3 x nrow x nx1 own cells, the nrow rows next to the face;
//!   fine -> coarse: (nx3/2) x nrow_fc x (nx1/2) restricted cells (2*nrow_fc fine rows);
//!   coarse -> fine: (nx3/2+2) x nrow_cf x (nx1/2+2) coarse cells of the fine block's
//!                   quadrant, with one ghost cell on every side for the limited slopes
//!                   (nrow_cf = nrow/2 + 2: the row across the face is a ghost row).

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"
#include "orbital_advection.hpp"
#include "mhd/mhd.hpp"
#include "remap_fluxes.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \fn OAFaceKind
//! \brief kind of the x2 neighbour of MeshBlock m on face f: 0 same level, 1 finer,
//! 2 coarser, -1 none (physical boundary). q returns the sub-face slot (0..3) of a same
//! or coarser neighbour.

template <typename NV>
KOKKOS_INLINE_FUNCTION
int OAFaceKind(const NV &nghbr, const int mylev, const int m, const int f, int &q) {
  int kind = -1;
  q = 0;
  const int base = (f == 0) ? 8 : 12;
  for (int s=0; s<4; ++s) {
    if (nghbr(m, base+s).gid >= 0) {
      int nlev = nghbr(m, base+s).lev;
      if (nlev == mylev) {
        kind = 0; q = s;
      } else if (nlev > mylev) {
        kind = 1;
      } else {
        kind = 2; q = s;
      }
    }
  }
  return kind;
}
} // namespace

//----------------------------------------------------------------------------------------
// OrbitalAdvectionCC derived class constructor:

OrbitalAdvectionCC::OrbitalAdvectionCC(MeshBlockPack *pp, ParameterInput *pin, int nv) :
    OrbitalAdvection(pp, pin), nvar_(nv) {
  int nmb = std::max((pp->nmb_thispack), (pp->pmesh->nmb_maxperrank));
  auto &indcs = pp->pmesh->mb_indcs;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const bool multilevel = pp->pmesh->multilevel;

  // Rows communicated across a same-level face: the largest shift plus the 3 cells of the
  // PPM remap stencil, but never more than one MeshBlock (all data must come from the
  // immediate x2 neighbour). RecvAndUnpackCC checks the actual shift every step.
  nrow = std::min(nx2, maxjshift + 3);
  nrow = pin->GetOrAddInteger("shearing_box", "oa_nrow", nrow);
  nrow = std::max(1, std::min(nx2, nrow));
  if (multilevel) {
    nrow = std::min(nx2, nrow + (nrow % 2));     // even: whole coarse cells
    nrow_fc = std::min(nrow, nx2/2);             // a fine block holds nx2/2 coarse rows
    nrow_cf = nrow/2 + 2;
  } else {
    nrow_fc = 0;
    nrow_cf = 0;
  }
  sz_same = nx3*nrow*nx1;
  sz_fc = multilevel ? (nx3/2)*nrow_fc*(nx1/2) : 0;
  sz_cf = multilevel ? (nx3/2 + 2)*nrow_cf*(nx1/2 + 2) : 0;
  sz_flx = multilevel ? (nx3/2)*(nx1/2) : 0;
  sz_face = nv*std::max(sz_same, 4*std::max(sz_fc, sz_cf));
  flux_correct = multilevel &&
                 pin->GetOrAddBoolean("shearing_box", "oa_flux_correct", true);

  Kokkos::realloc(sbuf_oa, nmb, 2, sz_face);
  Kokkos::realloc(rbuf_oa, nmb, 2, sz_face);
  Kokkos::realloc(sflx_oa, nmb, 2, std::max(1, nv*sz_flx));
  Kokkos::realloc(rflx_oa, nmb, 2, std::max(1, 4*nv*sz_flx));
  if (multilevel) {
    Kokkos::realloc(gflx, nmb, 2, nv, nx3, nx1);
  } else {
    Kokkos::realloc(gflx, 1, 2, nv, 1, 1);
  }

#if MPI_PARALLEL_ENABLED
  nreq = 8*nmb;
  sreq = new MPI_Request[nreq];
  rreq = new MPI_Request[nreq];
  sreq_flx = new MPI_Request[nreq];
  rreq_flx = new MPI_Request[nreq];
  for (int n=0; n<nreq; ++n) {
    sreq[n] = MPI_REQUEST_NULL;
    rreq[n] = MPI_REQUEST_NULL;
    sreq_flx[n] = MPI_REQUEST_NULL;
    rreq_flx[n] = MPI_REQUEST_NULL;
  }
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_orb_flx);
#endif
}

//----------------------------------------------------------------------------------------
// OrbitalAdvectionCC destructor

OrbitalAdvectionCC::~OrbitalAdvectionCC() {
#if MPI_PARALLEL_ENABLED
  delete [] sreq;
  delete [] rreq;
  delete [] sreq_flx;
  delete [] rreq_flx;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::PackAndSendCC()
//! \brief Pack cell-centered variables into boundary buffers and send to the x2-face
//! neighbours for the orbital advection step, as same-level rows, restricted rows (to a
//! coarser neighbour) or a coarse box for prolongation (to a finer neighbour).
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)

TaskStatus OrbitalAdvectionCC::PackAndSendCC(DvceArray5D<Real> &a) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sbuf_oa;
  auto &rbuf = rbuf_oa;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1, nx3 = indcs.nx3;
  const int nrow_ = nrow, nrow_fc_ = nrow_fc, nrow_cf_ = nrow_cf;
  const int sz_same_ = sz_same, sz_fc_ = sz_fc, sz_cf_ = sz_cf;

  // Outer loop over (# of MeshBlocks)*(8 x2 sub-faces)*(# of variables)
  int nmnv = nmb*8*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("oa-pack", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(8*nvar);
    const int s = (tmember.league_rank() - m*(8*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(8*nvar) - s*nvar);
    const int f = s/4;          // 0: inner x2 face, 1: outer x2 face
    const int q = s - 4*f;      // sub-face
    const int nn = 8 + s;       // index in nghbr view

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,nn).gid < 0) return;

    const int mylev = mblev.d_view(m);
    const int nlev = nghbr.d_view(m,nn).lev;
    // index of recv'ing (destination) MB, its face and sub-face: MB IDs are stored
    // sequentially in MeshBlockPacks, so array index equals (target_id - first_id)
    const int dm = nghbr.d_view(m,nn).gid - mbgid.d_view(0);
    const int ds = nghbr.d_view(m,nn).dest - 8;
    const int df = ds/4;
    const int dq = ds - 4*df;
    const bool same_rank = (nghbr.d_view(m,nn).rank == my_rank);

    if (nlev == mylev) {
      // same level: nrow rows next to face f
      const int jl = (f == 0) ? js : (je - nrow_ + 1);
      const int nji = nrow_*nx1;
      const int nkji = nx3*nji;
      const int soff = q*nvar*sz_same_ + v*sz_same_;
      const int doff = dq*nvar*sz_same_ + v*sz_same_;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/nx1;
        int i = idx - k*nji - j*nx1;
        Real val = a(m,v,k+ks,j+jl,i+is);
        if (same_rank) {
          rbuf(dm,df,doff+idx) = val;
        } else {
          sbuf(m,f,soff+idx) = val;
        }
      });
    } else if (nlev < mylev) {
      // coarser neighbour: restrict 2*nrow_fc rows next to face f
      const int jl = (f == 0) ? js : (je - 2*nrow_fc_ + 1);
      const int cnx1 = nx1/2;
      const int nji = nrow_fc_*cnx1;
      const int nkji = (nx3/2)*nji;
      const int soff = q*nvar*sz_fc_ + v*sz_fc_;
      const int doff = dq*nvar*sz_fc_ + v*sz_fc_;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int kc = idx/nji;
        int jc = (idx - kc*nji)/cnx1;
        int ic = idx - kc*nji - jc*cnx1;
        int k = ks + 2*kc, j = jl + 2*jc, i = is + 2*ic;
        Real val = 0.125*(a(m,v,k  ,j  ,i) + a(m,v,k  ,j  ,i+1)
                        + a(m,v,k  ,j+1,i) + a(m,v,k  ,j+1,i+1)
                        + a(m,v,k+1,j  ,i) + a(m,v,k+1,j  ,i+1)
                        + a(m,v,k+1,j+1,i) + a(m,v,k+1,j+1,i+1));
        if (same_rank) {
          rbuf(dm,df,doff+idx) = val;
        } else {
          sbuf(m,f,soff+idx) = val;
        }
      });
    } else {
      // finer neighbour in quadrant q = fx + 2*fz: coarse box with one ghost cell on
      // every side, rows js-1..js+nrow/2 (f=0) or je-nrow/2..je+1 (f=1)
      const int fx = q & 1, fz = q >> 1;
      const int il = is + fx*(nx1/2) - 1;
      const int kl = ks + fz*(nx3/2) - 1;
      const int jl = (f == 0) ? (js - 1) : (je - nrow_/2);
      const int bnx1 = nx1/2 + 2;
      const int nji = nrow_cf_*bnx1;
      const int nkji = (nx3/2 + 2)*nji;
      const int soff = q*nvar*sz_cf_ + v*sz_cf_;
      const int doff = dq*nvar*sz_cf_ + v*sz_cf_;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/bnx1;
        int i = idx - k*nji - j*bnx1;
        Real val = a(m,v,k+kl,j+jl,i+il);
        if (same_rank) {
          rbuf(dm,df,doff+idx) = val;
        } else {
          sbuf(m,f,soff+idx) = val;
        }
      });
    }
  }); // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int s=0; s<8; ++s) {
      int nn = 8 + s;
      if (nghbr.h_view(m,nn).gid >= 0) { // neighbor exists and not a physical bndry
        int drank = nghbr.h_view(m,nn).rank;
        if (drank != my_rank) {
          int f = s/4, q = s - 4*f;
          int nlev = nghbr.h_view(m,nn).lev, mylev = mblev.h_view(m);
          int sz = (nlev == mylev) ? sz_same : ((nlev < mylev) ? sz_fc : sz_cf);
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,nn).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, nghbr.h_view(m,nn).dest);
          auto send_ptr = Kokkos::subview(sbuf, m, f,
                              std::make_pair(q*nvar*sz, (q+1)*nvar*sz));
          int ierr = MPI_Isend(send_ptr.data(), nvar*sz, MPI_ATHENA_REAL, drank, tag,
                               comm_orb_advect, &(sreq[8*m+s]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \!fn void OrbitalAdvectionCC::RecvAndUnpackCC()
//! \brief Receive and unpack boundary buffers for CC variables with orbital advection,
//! and apply shift in x2- (y-) direction across entire MeshBlock by applying both an
//! integer shift and a fractional offset to input array a. With mesh refinement, also
//! stores the remap "fluxes" through the two x2 faces of every MeshBlock (gflx) for the
//! flux correction across level jumps.

TaskStatus OrbitalAdvectionCC::RecvAndUnpackCC(DvceArray5D<Real> &a,
                                               ReconstructionMethod rcon) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  auto &rbuf = rbuf_oa;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<8*nmb; ++n) {
    int test;
    int ierr = MPI_Test(&(rreq[n]), &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so unpack and apply shift

  // Guard against shifts deeper than the communicated rows (see RowsAvailable). The
  // shift in cells is qshear*omega0*|x1|*dt/dx2, checked here with this step's dt;
  // MaxTimeStep() normally keeps dt below the limit.
  {
    auto &mbsize_h = pmy_pack->pmb->mb_size;
    Real qodt = qshear*omega0*(pmy_pack->pmesh->dt);
    for (int m=0; m<nmb; ++m) {
      Real xmax = std::max(fabs(mbsize_h.h_view(m).x1min),
                           fabs(mbsize_h.h_view(m).x1max));
      int joff = static_cast<int>(qodt*xmax/(mbsize_h.h_view(m).dx2));
      int avail = RowsAvailable(m);
      if (joff + 3 > avail) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Orbital advection shift (" << joff << " cells) plus the "
          << "3-cell remap stencil exceeds the " << avail << " rows available from the "
          << "x2 neighbour of MeshBlock gid=" << pmy_pack->pmb->mb_gid.h_view(m)
          << " (level " << mblev.h_view(m) << ", x1 in [" << mbsize_h.h_view(m).x1min
          << "," << mbsize_h.h_view(m).x1max << "]). Rows available: nrow=" << nrow
          << " (<shearing_box> oa_nrow, at most the MeshBlock nx2), or nx2/2="
          << nrow_fc << " next to a finer x2 neighbour. Use larger MeshBlocks in x2 "
          << "or a smaller cfl_number." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx3 = indcs.nx3;
  const int nrow_ = nrow, nrow_fc_ = nrow_fc, nrow_cf_ = nrow_cf;
  const int sz_same_ = sz_same, sz_fc_ = sz_fc, sz_cf_ = sz_cf;
  const bool multilevel = pmy_pack->pmesh->multilevel;
  auto &gflx_ = gflx;
  int jfs = nrow;
  int jfe = jfs + indcs.nx2 - 1;
  int nfx = indcs.nx2 + 2*nrow;

  auto &mbsize = pmy_pack->pmb->mb_size;
  Real &dt = pmy_pack->pmesh->dt;
  Real qo = qshear*omega0;

  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nfx) * 2;
  par_for_outer("oa-unpk",DevExeSpace(),scr_size,scr_lvl,0,(nmb-1),0,(nvar-1),ks,ke,is,ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int i) {
    ScrArray1D<Real> a_(member.team_scratch(scr_lvl), nfx); // 1D slice of data
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nfx); // "flux" at faces

    Real &x1min = mbsize.d_view(m).x1min;
    Real &x1max = mbsize.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real yshear = -(qo)*x1v*dt;
    int joffset = static_cast<int>(yshear/(mbsize.d_view(m).dx2));

    // kind of the two x2 neighbours
    const int mylev = mblev.d_view(m);
    int qf[2];
    int kind[2];
    kind[0] = OAFaceKind(nghbr.d_view, mylev, m, 0, qf[0]);
    kind[1] = OAFaceKind(nghbr.d_view, mylev, m, 1, qf[1]);

    // value of row t (ascending j) of the column beyond face f, at this block's
    // resolution, from the received message
    auto load = [&](const int f, const int t) -> Real {
      if (kind[f] == 0) {
        return rbuf(m,f, qf[f]*nvar*sz_same_ + n*sz_same_ + (k-ks)*nrow_*nx1
                         + t*nx1 + (i-is));
      } else if (kind[f] == 1) {
        // finer neighbours: restricted rows, quadrant from (i,k)
        const int cnx1 = nx1/2, cnx3 = nx3/2;
        const int fx = (i-is)/cnx1, fz = (k-ks)/cnx3;
        const int ic = (i-is) - fx*cnx1, kc = (k-ks) - fz*cnx3;
        const int tc = (f == 0) ? (t - (nrow_ - nrow_fc_)) : t;
        if (tc < 0 || tc >= nrow_fc_) return 0.0;   // never read (guarded above)
        return rbuf(m,f, (fx + 2*fz)*nvar*sz_fc_ + n*sz_fc_ + kc*nrow_fc_*cnx1
                         + tc*cnx1 + ic);
      } else if (kind[f] == 2) {
        // coarser neighbour: prolongate the coarse box as in ProlongCC
        const int d = (f == 0) ? (nrow_ - 1 - t) : t;    // distance from the face
        const int jj = (f == 0) ? (nrow_/2 - d/2) : (d/2 + 1);
        const Real s2 = (f == 0) ? ((d % 2 == 0) ? 1.0 : -1.0)
                                 : ((d % 2 == 0) ? -1.0 : 1.0);
        const int ii = (i-is)/2 + 1, kk = (k-ks)/2 + 1;
        const Real s1 = (((i-is) % 2) == 0) ? -1.0 : 1.0;
        const Real s3 = (((k-ks) % 2) == 0) ? -1.0 : 1.0;
        const int bnx1 = nx1/2 + 2;
        const int bnji = nrow_cf_*bnx1;
        const int base = qf[f]*nvar*sz_cf_ + n*sz_cf_;
        auto c = [&](const int dk, const int dj, const int di) -> Real {
          return rbuf(m,f, base + (kk+dk)*bnji + (jj+dj)*bnx1 + (ii+di));
        };
        Real c0 = c(0,0,0);
        Real dl = c0 - c(0,0,-1);
        Real dr = c(0,0,1) - c0;
        Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
        dl = c0 - c(0,-1,0);
        dr = c(0,1,0) - c0;
        Real dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
        dl = c0 - c(-1,0,0);
        dr = c(1,0,0) - c0;
        Real dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
        return c0 + s1*dvar1 + s2*dvar2 + s3*dvar3;
      }
      return 0.0;
    };

    // Load scratch array with no shift
    par_for_inner(member, 0, (nfx-1), [&](const int jf) {
      if (jf < jfs) {
        // Load from L boundary buffer
        a_(jf) = load(0, jf);
      } else if (jf <= jfe) {
        // Load from conserved variables themselves (addressed with j=jf-jfs+js)
        a_(jf) = a(m,n,k,jf-jfs+js,i);
      } else {
        // Load from R boundary buffer
        a_(jf) = load(1, jf-(jfe+1));
      }
    });
    member.team_barrier();

    // Compute "fluxes" at shifted cell faces
    Real epsi = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, a_, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, a_, flx);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
      case ReconstructionMethod::teno:
      case ReconstructionMethod::wenoz:
      case ReconstructionMethod::wenoz_mc:
        PPMX_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, a_, flx);
        break;
      default:
        break;
    }
    member.team_barrier();

    // With refinement, store the total remap "flux" G through each x2 face of the block:
    // a_new(jf) = a_(jf) - (G(jf+1) - G(jf)), where G(jf) is the mass (etc.) moved in +x2
    // through face jf: the |joffset| whole cells it passes plus the fractional flux.
    if (multilevel) {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        for (int f=0; f<2; ++f) {
          const int jf = (f == 0) ? jfs : (jfe + 1);
          Real g = flx(jf - joffset);
          if (joffset > 0) {
            for (int l=jf-joffset; l<jf; ++l) {g += a_(l);}
          } else if (joffset < 0) {
            for (int l=jf; l<jf-joffset; ++l) {g -= a_(l);}
          }
          gflx_(m,f,n,k-ks,i-is) = g;
        }
      });
    }

    // Update CC variables with both integer shift (from a_) and a conservative remap
    // for the remaining fraction of a cell using upwind "fluxes"
    par_for_inner(member, js, je, [&](const int j) {
      int jf = j-js + jfs;
      a(m,n,k,j,i) = a_(jf-joffset) - (flx(jf+1-joffset) - flx(jf-joffset));
    });
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn int OrbitalAdvectionCC::RowsAvailable()
//! \brief Rows (in MeshBlock m's cells) its x2 neighbours supply to the remap. The remap
//! of a cell reads up to |shift| + 3 cells upstream (PPM stencil), all of which must come
//! from the immediate x2 neighbour: a same-level or coarser neighbour provides nrow rows,
//! a finer one only nrow_fc coarse rows (its whole block is nx2/2 coarse rows).

int OrbitalAdvectionCC::RowsAvailable(int m) {
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int avail = nrow;
  for (int f=0; f<2; ++f) {
    int q;
    if (OAFaceKind(nghbr.h_view, mblev.h_view(m), m, f, q) == 1) {
      avail = std::min(avail, nrow_fc);
    }
  }
  return avail;
}

//----------------------------------------------------------------------------------------
//! \fn Real OrbitalAdvectionCC::MaxTimeStep()
//! \brief Largest timestep for which the integer shift int(q*Omega*|x1|*dt/dx2) of every
//! MeshBlock stays within RowsAvailable() - 3, i.e. q*Omega*max|x1|*dt/dx2 < avail - 2
//! (with a 0.1% margin). Called by Mesh::NewTimeStep (after any AMR update, so on the
//! current mesh). It binds only when the shift is large: e.g. coarse 8^3 blocks next to
//! a finer x2 neighbour allow one coarse cell per step. Not multiplied by cfl_number.

Real OrbitalAdvectionCC::MaxTimeStep() {
  Real dtmin = std::numeric_limits<float>::max();
  if (!(pmy_pack->pmesh->three_d)) return dtmin;
  auto &mbsize_h = pmy_pack->pmb->mb_size;
  Real qo = fabs(qshear*omega0);
  for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
    Real xmax = std::max(fabs(mbsize_h.h_view(m).x1min),
                         fabs(mbsize_h.h_view(m).x1max));
    int avail = RowsAvailable(m);
    if (qo*xmax > 0.0 && avail > 2) {
      dtmin = std::min(dtmin, 0.999*(avail - 2)*(mbsize_h.h_view(m).dx2)/(qo*xmax));
    }
  }
  return dtmin;
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::SendFluxCC()
//! \brief With refinement: every MeshBlock whose x2 neighbour is coarser restricts the
//! remap fluxes through that face (sum over the 2x2 fine columns of a coarse column,
//! times the volume ratio 1/8) and sends them to the coarse neighbour.

TaskStatus OrbitalAdvectionCC::SendFluxCC() {
  if (!flux_correct) return TaskStatus::complete;
  int nmb = pmy_pack->nmb_thispack;
  int nvar = nvar_;
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sflx_oa;
  auto &rbuf = rflx_oa;
  auto &gflx_ = gflx;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int cnx1 = indcs.nx1/2, cnx3 = indcs.nx3/2;
  const int sz_flx_ = sz_flx;

  par_for("oa-flxsend", DevExeSpace(), 0, (nmb-1), 0, 1, 0, (nvar-1), 0, (cnx3-1),
          0, (cnx1-1),
  KOKKOS_LAMBDA(const int m, const int f, const int v, const int kc, const int ic) {
    int q;
    if (OAFaceKind(nghbr.d_view, mblev.d_view(m), m, f, q) != 2) return;
    const int nn = 8 + 4*f + q;
    Real val = 0.125*(gflx_(m,f,v,2*kc  ,2*ic) + gflx_(m,f,v,2*kc  ,2*ic+1)
                    + gflx_(m,f,v,2*kc+1,2*ic) + gflx_(m,f,v,2*kc+1,2*ic+1));
    const int idx = v*sz_flx_ + kc*cnx1 + ic;
    if (nghbr.d_view(m,nn).rank == my_rank) {
      const int dm = nghbr.d_view(m,nn).gid - mbgid.d_view(0);
      const int ds = nghbr.d_view(m,nn).dest - 8;
      const int df = ds/4;
      const int dq = ds - 4*df;
      rbuf(dm,df,dq*nvar*sz_flx_ + idx) = val;
    } else {
      sbuf(m,f,idx) = val;
    }
  });

#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int f=0; f<2; ++f) {
      int q;
      if (OAFaceKind(nghbr.h_view, mblev.h_view(m), m, f, q) != 2) continue;
      const int nn = 8 + 4*f + q;
      int drank = nghbr.h_view(m,nn).rank;
      if (drank != my_rank) {
        int lid = nghbr.h_view(m,nn).gid - pmy_pack->pmesh->gids_eachrank[drank];
        int tag = CreateBvals_MPI_Tag(lid, nghbr.h_view(m,nn).dest);
        auto send_ptr = Kokkos::subview(sbuf, m, f, std::make_pair(0, nvar*sz_flx));
        int ierr = MPI_Isend(send_ptr.data(), nvar*sz_flx, MPI_ATHENA_REAL, drank, tag,
                             comm_orb_flx, &(sreq_flx[8*m + 4*f + q]));
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends of orbital-advection fluxes"
       << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::RecvAndCorrectFluxCC()
//! \brief With refinement: every MeshBlock whose x2 neighbours are finer replaces its own
//! remap flux through that face by the restricted fine fluxes. Only the row next to the
//! face changes: a(js) += G_fine - G_own on the inner face, a(je) += G_own - G_fine on
//! the outer face.

TaskStatus OrbitalAdvectionCC::RecvAndCorrectFluxCC(DvceArray5D<Real> &a) {
  if (!flux_correct) return TaskStatus::complete;
  int nmb = pmy_pack->nmb_thispack;
#if MPI_PARALLEL_ENABLED
  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<8*nmb; ++n) {
    int test;
    int ierr = MPI_Test(&(rreq_flx[n]), &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {bflag = true;}
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bflag) {return TaskStatus::incomplete;}
#endif

  int nvar = nvar_;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = rflx_oa;
  auto &gflx_ = gflx;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, je = indcs.je, ks = indcs.ks;
  const int cnx1 = indcs.nx1/2, cnx3 = indcs.nx3/2;
  const int sz_flx_ = sz_flx;

  // the two faces in separate kernels, so a block with nx2 = 1 could not race
  for (int f=0; f<2; ++f) {
    par_for("oa-flxcorr", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), ks, indcs.ke, is,
            indcs.ie,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int i) {
      int q;
      if (OAFaceKind(nghbr.d_view, mblev.d_view(m), m, f, q) != 1) return;
      const int fx = (i-is)/cnx1, fz = (k-ks)/cnx3;
      const int ic = (i-is) - fx*cnx1, kc = (k-ks) - fz*cnx3;
      Real gfine = rbuf(m,f,(fx + 2*fz)*nvar*sz_flx_ + v*sz_flx_ + kc*cnx1 + ic);
      Real gown = gflx_(m,f,v,k-ks,i-is);
      if (f == 0) {
        a(m,v,k,js,i) += gfine - gown;
      } else {
        a(m,v,k,je,i) += gown - gfine;
      }
    });
  }
  return TaskStatus::complete;
}
