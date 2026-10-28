//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection_fc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for face-centered (FC)
//! variables in the orbital advection step used with the shearing box. Data is shifted
//! by the appropriate offset during the recv/unpack step, so these functions both
//! communicate the data and perform the shift.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "shearing_box.hpp"
#include "orbital_advection.hpp"
#include "mhd/mhd.hpp"
#include "bvals/bvals.hpp"
#include "remap_fluxes.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \fn OAFaceKindB
//! \brief kind of the x2 neighbour of MeshBlock m on face f: 0 same level, 1 finer,
//! 2 coarser, -1 none; q returns the sub-face slot of a same or coarser neighbour
//! (as OAFaceKind in orbital_advection_cc.cpp)

template <typename NV>
KOKKOS_INLINE_FUNCTION
int OAFaceKindB(const NV &nghbr, const int mylev, const int m, const int f, int &q) {
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
// OrbitalAdvectionFC derived class constructor:

OrbitalAdvectionFC::OrbitalAdvectionFC(MeshBlockPack *pp, ParameterInput *pin) :
  OrbitalAdvection(pp, pin),
  eoa("eoa",1,1,1,1) {
  // Initialize boundary buffers. The remap reads up to |shift| + 3 rows of the x2
  // neighbour, and the cell-centred remap (OrbitalAdvectionCC, same shift) keeps that
  // within its nrow <= nx2 rows through the timestep limit; send at least as many rows,
  // but never more than one MeshBlock.
  int nmb = std::max((pp->nmb_thispack), (pp->pmesh->nmb_maxperrank));
  auto &indcs = pp->pmesh->mb_indcs;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const bool multilevel = pp->pmesh->multilevel;
  nrowb = std::max(indcs.ng + maxjshift + 1, maxjshift + 3);
  if (pin->DoesParameterExist("shearing_box", "oa_nrow")) {
    nrowb = std::max(nrowb, pin->GetInteger("shearing_box", "oa_nrow"));
  }
  nrowb = std::min(nrowb, nx2);
  if (multilevel) {
    nrowb = std::min(nx2, nrowb + (nrowb % 2));     // whole coarse cells
    nrowb_fc = std::min(nrowb, nx2/2);
    nrowb_cf = nrowb/2 + 2;
  } else {
    nrowb_fc = 0;
    nrowb_cf = 0;
  }
  // Messages per component (0: B3 on x3 faces, 1: B1 on x1 faces), with one footprint
  // for both: (nx3+1) x rows x (nx1+1) at the receiver's resolution. B3 at i = ie+1 and
  // B1 at k = ke+1 are not used. A coarse box carries one more layer on every side.
  szb_same = (nx3 + 1)*nrowb*(nx1 + 1);
  szb_fc = multilevel ? (nx3/2 + 1)*nrowb_fc*(nx1/2 + 1) : 0;
  szb_cf = multilevel ? (nx3/2 + 3)*nrowb_cf*(nx1/2 + 3) : 0;
  szb_face = 2*std::max(szb_same, 4*std::max(szb_fc, szb_cf));
  Kokkos::realloc(sbufb, nmb, 2, szb_face);
  Kokkos::realloc(rbufb, nmb, 2, szb_face);
#if MPI_PARALLEL_ENABLED
  nreqb = 8*nmb;
  sreqb = new MPI_Request[nreqb];
  rreqb = new MPI_Request[nreqb];
  for (int n=0; n<nreqb; ++n) {
    sreqb[n] = MPI_REQUEST_NULL;
    rreqb[n] = MPI_REQUEST_NULL;
  }
#endif
  int ncells1, ncells2, ncells3;

  // Allocate memory for electric fields (for every MeshBlock the pack may hold after AMR)
  ncells1 = indcs.nx1 + 2*(indcs.ng);
  ncells2 = indcs.nx2 + 2*(indcs.ng);
  ncells3 = indcs.nx3 + 2*(indcs.ng);
  Kokkos::realloc(emfx,nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(emfz,nmb,ncells3,ncells2,ncells1);

  // With mesh refinement, the EMFs of the remap are corrected at level jumps like the
  // stage EMFs, through a private flux-correction instance (own buffers, communicator)
  flux_correct = pp->pmesh->multilevel &&
                 pin->GetOrAddBoolean("shearing_box", "oa_flux_correct", true);
  if (flux_correct) {
    Kokkos::realloc(eoa.x1e, nmb, ncells3+1, ncells2+1, ncells1);
    Kokkos::realloc(eoa.x2e, nmb, ncells3+1, ncells2, ncells1+1);
    Kokkos::realloc(eoa.x3e, nmb, ncells3, ncells2+1, ncells1+1);
    pbval_oa = new MeshBoundaryValuesFC(pp, pin);
    pbval_oa->InitializeBuffers(3);
  }
}

//----------------------------------------------------------------------------------------
// OrbitalAdvectionFC destructor

OrbitalAdvectionFC::~OrbitalAdvectionFC() {
  if (pbval_oa != nullptr) {delete pbval_oa;}
#if MPI_PARALLEL_ENABLED
  delete [] sreqb;
  delete [] rreqb;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::PackAndSendFC()
//! \brief Pack B3 and B1 next to the x2 faces into boundary buffers and send them to the
//! x2 neighbours for the orbital advection step: same-level rows, rows area-averaged to
//! a coarser neighbour, or a coarse box (one extra layer on every side) to a finer one.
//! Layout per (MeshBlock, face): sub-face q at q*2*(size), component c at c*(size), then
//! (k, row, i) with i fastest and rows in ascending j.

TaskStatus OrbitalAdvectionFC::PackAndSendFC(DvceFaceFld4D<Real> &b) {
  int nmb = pmy_pack->nmb_thispack;
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sbufb;
  auto &rbuf = rbufb;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, je = indcs.je, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx3 = indcs.nx3;
  const int nrowb_ = nrowb, nrowb_fc_ = nrowb_fc, nrowb_cf_ = nrowb_cf;
  const int szs = szb_same, szf = szb_fc, szc = szb_cf;

  // Outer loop over (# of MeshBlocks)*(8 x2 sub-faces)*(2 components)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmb*8*2, Kokkos::AUTO);
  Kokkos::parallel_for("oa-packB", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank()/16;
    const int s = (tmember.league_rank() - m*16)/2;
    const int c = tmember.league_rank() - m*16 - s*2;     // 0: B3, 1: B1
    const int f = s/4, q = s - 4*f, nn = 8 + s;
    if (nghbr.d_view(m,nn).gid < 0) return;

    const int mylev = mblev.d_view(m);
    const int nlev = nghbr.d_view(m,nn).lev;
    const int dm = nghbr.d_view(m,nn).gid - mbgid.d_view(0);
    const int ds = nghbr.d_view(m,nn).dest - 8;
    const int df = ds/4, dq = ds - 4*df;
    const bool same_rank = (nghbr.d_view(m,nn).rank == my_rank);
    auto &fld = (c == 0) ? b.x3f : b.x1f;

    if (nlev == mylev) {
      // same level: nrowb rows next to face f
      const int jl = (f == 0) ? js : (je - nrowb_ + 1);
      const int ni = nx1 + 1, nji = nrowb_*ni, nkji = (nx3 + 1)*nji;
      const int soff = q*2*szs + c*szs, doff = dq*2*szs + c*szs;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji, j = (idx - k*nji)/ni, i = idx - k*nji - j*ni;
        Real val = fld(m, k+ks, j+jl, i+is);
        if (same_rank) {rbuf(dm,df,doff+idx) = val;} else {sbuf(m,f,soff+idx) = val;}
      });
    } else if (nlev < mylev) {
      // coarser neighbour: average 2 x 2 fine faces onto each coarse face
      const int jl = (f == 0) ? js : (je - 2*nrowb_fc_ + 1);
      const int ni = nx1/2 + 1, nji = nrowb_fc_*ni, nkji = (nx3/2 + 1)*nji;
      const int soff = q*2*szf + c*szf, doff = dq*2*szf + c*szf;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int kc = idx/nji, jc = (idx - kc*nji)/ni, ic = idx - kc*nji - jc*ni;
        int k = ks + 2*kc, j = jl + 2*jc, i = is + 2*ic;
        Real val;
        if (c == 0) {   // B3 on the x3 face k: average over (i, i+1) x (j, j+1)
          val = 0.25*(fld(m,k,j,i) + fld(m,k,j,i+1) + fld(m,k,j+1,i) + fld(m,k,j+1,i+1));
        } else {        // B1 on the x1 face i: average over (j, j+1) x (k, k+1)
          val = 0.25*(fld(m,k,j,i) + fld(m,k,j+1,i) + fld(m,k+1,j,i) + fld(m,k+1,j+1,i));
        }
        if (same_rank) {rbuf(dm,df,doff+idx) = val;} else {sbuf(m,f,soff+idx) = val;}
      });
    } else {
      // finer neighbour in quadrant q = fx + 2*fz: coarse box, one extra layer around
      const int fx = q & 1, fz = q >> 1;
      const int il = is + fx*(nx1/2) - 1, kl = ks + fz*(nx3/2) - 1;
      const int jl = (f == 0) ? (js - 1) : (je - nrowb_/2);
      const int ni = nx1/2 + 3, nji = nrowb_cf_*ni, nkji = (nx3/2 + 3)*nji;
      const int soff = q*2*szc + c*szc, doff = dq*2*szc + c*szc;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji, j = (idx - k*nji)/ni, i = idx - k*nji - j*ni;
        Real val = fld(m, k+kl, j+jl, i+il);
        if (same_rank) {rbuf(dm,df,doff+idx) = val;} else {sbuf(m,f,soff+idx) = val;}
      });
    }
  });

#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int s=0; s<8; ++s) {
      int nn = 8 + s;
      if (nghbr.h_view(m,nn).gid >= 0) {
        int drank = nghbr.h_view(m,nn).rank;
        if (drank != my_rank) {
          int f = s/4, q = s - 4*f;
          int nlev = nghbr.h_view(m,nn).lev, mylev = mblev.h_view(m);
          int sz = (nlev == mylev) ? szb_same : ((nlev < mylev) ? szb_fc : szb_cf);
          int lid = nghbr.h_view(m,nn).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, nghbr.h_view(m,nn).dest);
          auto send_ptr = Kokkos::subview(sbuf, m, f,
                              std::make_pair(q*2*sz, (q+1)*2*sz));
          int ierr = MPI_Isend(send_ptr.data(), 2*sz, MPI_ATHENA_REAL, drank, tag,
                               comm_orb_advect, &(sreqb[8*m+s]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \!fn void OrbitalAdvectionFC::RecvAndUnpackFC()
//! \brief Receive and unpack boundary buffers for FC fields with orbital advection, and
//! apply shift in x2- (y-) direction across entire MeshBlock. Since CT is required to
//! update fields, the algorithm used here is different from that used for CC variables in
//! RecvAndUnpackCC(). Here an effective electric field is computed including both the
//! integer and fractional cell shifts. These fields are then used to update B using CT.
//! The fields themselves are not directly remapped like the CC variables.

TaskStatus OrbitalAdvectionFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b0,
                                             ReconstructionMethod rcon) {
  int nmb = pmy_pack->nmb_thispack;
  auto &rbuf = rbufb;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<8*nmb; ++n) {
    int test;
    int ierr = MPI_Test(&(rreqb[n]), &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {bflag = true;}
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

  //----- STEP 2: buffers have all completed, so unpack and compute effective EMF

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  int jfs = nrowb;
  int jfe = jfs + indcs.nx2 - 1;
  int nfx = indcs.nx2 + 2*nrowb;
  const int nx3 = indcs.nx3;
  const int nrowb_ = nrowb, nrowb_fc_ = nrowb_fc, nrowb_cf_ = nrowb_cf;
  const int szs = szb_same, szf = szb_fc, szc = szb_cf;

  auto &mbsize = pmy_pack->pmb->mb_size;
  Real &dt = pmy_pack->pmesh->dt;
  Real qo = qshear*omega0;

  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nfx) * 2;
  auto &emfx_ = emfx;
  auto &emfz_ = emfz;
  par_for_outer("oa-unB",DevExeSpace(),scr_size,scr_lvl,0,(nmb-1),0,1,ks,ke+1,is,ie+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int v, const int k, const int i) {
    ScrArray1D<Real> b0_(member.team_scratch(scr_lvl), nfx); // 1D slice of data
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nfx); // "flux" at faces

    Real &x1min = mbsize.d_view(m).x1min;
    Real &x1max = mbsize.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real x1;
    if (v==0) {
      // B3 located at x1-cell centers
      x1 = CellCenterX(i-is, nx1, x1min, x1max);
    } else {
      // B1 located at x1-cell faces
      x1 = LeftEdgeX(i-is, nx1, x1min, x1max);
    }
    Real yshear = -(qo)*x1*dt;
    int joffset = static_cast<int>(yshear/(mbsize.d_view(m).dx2));

    // kind of the two x2 neighbours
    const int mylev = mblev.d_view(m);
    int qf[2], kind[2];
    kind[0] = OAFaceKindB(nghbr.d_view, mylev, m, 0, qf[0]);
    kind[1] = OAFaceKindB(nghbr.d_view, mylev, m, 1, qf[1]);

    // row t (ascending j) of the column beyond face f at this block's resolution;
    // v = 0: B3 (x3 face k, x1 cell i), v = 1: B1 (x1 face i, x3 cell k)
    auto load = [&](const int f, const int t) -> Real {
      if (kind[f] == 0) {
        return rbuf(m,f, qf[f]*2*szs + v*szs + (k-ks)*nrowb_*(nx1+1) + t*(nx1+1)
                         + (i-is));
      } else if (kind[f] == 1) {
        // finer neighbours: area-averaged rows; quadrant from (i,k), the face on a
        // quadrant boundary taken from the upper quadrant (both copies are equal)
        const int cnx1 = nx1/2, cnx3 = nx3/2;
        const int fx = ((i-is) < cnx1) ? 0 : 1, fz = ((k-ks) < cnx3) ? 0 : 1;
        const int ic = (i-is) - fx*cnx1, kc = (k-ks) - fz*cnx3;
        const int tc = (f == 0) ? (t - (nrowb_ - nrowb_fc_)) : t;
        if (tc < 0 || tc >= nrowb_fc_) return 0.0;   // never read (guarded)
        return rbuf(m,f, (fx + 2*fz)*2*szf + v*szf + kc*nrowb_fc_*(cnx1+1)
                         + tc*(cnx1+1) + ic);
      } else if (kind[f] == 2) {
        // coarser neighbour: limited-linear interpolation of the coarse faces. Along
        // the field's own direction a fine face either coincides with a coarse face or
        // lies midway between two; across it the minmod slopes of ProlongCC are used.
        const int d = (f == 0) ? (nrowb_ - 1 - t) : t;
        const int jj = (f == 0) ? (nrowb_/2 - d/2) : (d/2 + 1);
        const Real s2 = (f == 0) ? ((d % 2 == 0) ? 1.0 : -1.0)
                                 : ((d % 2 == 0) ? -1.0 : 1.0);
        const int bni = nx1/2 + 3, bnji = nrowb_cf_*bni;
        const int base = qf[f]*2*szc + v*szc;
        auto c = [&](const int kk, const int j2, const int ii) -> Real {
          return rbuf(m,f, base + kk*bnji + j2*bni + ii);
        };
        auto mm = [](const Real dl, const Real dr) -> Real {
          return 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
        };
        // value on the coarse face (kk, ii) of component v, moved to the fine position
        // across the face direction by the slopes in the two transverse directions
        auto face = [&](const int kk, const int ii, const Real sa) -> Real {
          Real c0 = c(kk,jj,ii);
          Real dy = mm(c0 - c(kk,jj-1,ii), c(kk,jj+1,ii) - c0);
          Real dt_ = (v == 0) ? mm(c0 - c(kk,jj,ii-1), c(kk,jj,ii+1) - c0)   // x for B3
                              : mm(c0 - c(kk-1,jj,ii), c(kk+1,jj,ii) - c0);  // z for B1
          return c0 + s2*dy + sa*dt_;
        };
        if (v == 1) {   // B1 at fine x1 face i, fine x3 cell k
          const int a = (i-is)/2, ii = a + 1, kk = (k-ks)/2 + 1;
          const Real s3 = (((k-ks) % 2) == 0) ? -1.0 : 1.0;
          if (((i-is) % 2) == 0) return face(kk, ii, s3);
          return 0.5*(face(kk, ii, s3) + face(kk, ii+1, s3));
        } else {        // B3 at fine x3 face k, fine x1 cell i
          const int a = (k-ks)/2, kk = a + 1, ii = (i-is)/2 + 1;
          const Real s1 = (((i-is) % 2) == 0) ? -1.0 : 1.0;
          if (((k-ks) % 2) == 0) return face(kk, ii, s1);
          return 0.5*(face(kk, ii, s1) + face(kk+1, ii, s1));
        }
      }
      return 0.0;
    };

    // Load scratch array with no shift
    par_for_inner(member, 0, (nfx-1), [&](const int jf) {
      if (jf < jfs) {
        // Load from L boundary buffer
        b0_(jf) = load(0, jf);
      } else if (jf <= jfe) {
        // Load from array itself (addressed with j=jf-jfs+js)
        if (v==0) {
          b0_(jf) = b0.x3f(m,k,(jf-jfs+js),i);
        } else if (v==1) {
          b0_(jf) = b0.x1f(m,k,(jf-jfs+js),i);
        }
      } else {
        // Load scratch arrays from R boundary buffer
        b0_(jf) = load(1, jf-(jfe+1));
      }
    });
    member.team_barrier();

    // Compute x2-fluxes at shifted cell faces
    Real epsi = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, b0_, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, b0_, flx);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
      case ReconstructionMethod::teno:
      case ReconstructionMethod::wenoz:
      case ReconstructionMethod::wenoz_mc:
        PPMX_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, b0_, flx);
        break;
      default:
        break;
    }
    member.team_barrier();

    // Compute emfx = -VyBz, which is at cell-center in x1-direction
    if (v==0) {
      par_for_inner(member, js, je+1, [&](const int j) {
        int jf = j-js + jfs;
        emfx_(m,k,j,i) = -flx(jf-joffset);
        // Sum integer offsets into effective EMFs
        for (int jj=1; jj<=joffset; jj++) {
          emfx_(m,k,j,i) -= b0_(jf-jj);
        }
        for (int jj=(joffset+1); jj<=0; jj++) {
          emfx_(m,k,j,i) += b0_(jf-jj);
        }
      });
      member.team_barrier();

    // Compute emfz =  VyBx, which is at cell-face in x1-direction
    } else if (v==1) {
      par_for_inner(member, js, je+1, [&](const int j) {
        int jf = j-js + jfs;
        emfz_(m,k,j,i) = flx(jf-joffset);
        // Sum integer offsets into effective EMFs
        for (int jj=1; jj<=joffset; jj++) {
          emfz_(m,k,j,i) += b0_(jf-jj);
        }
        for (int jj=(joffset+1); jj<=0; jj++) {
          emfz_(m,k,j,i) -= b0_(jf-jj);
        }
      });
      member.team_barrier();
    }
  });

  // Without refinement update B now; with it the EMFs are corrected first
  // (SendEMF_OA, RecvEMFAndUpdateB_OA)
  if (!flux_correct) {
    UpdateB(b0);
    return TaskStatus::complete;
  }

  // Store the EMFs as E*dt on the edges for the correction: emfx (E1) on x1 edges,
  // emfz (E3) on x3 edges; the remap has no E2.
  auto &eoa_ = eoa;
  par_for("oa-eoa1", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    eoa_.x1e(m,k,j,i) = emfx_(m,k,j,i)*mbsize.d_view(m).dx2;
  });
  par_for("oa-eoa3", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    eoa_.x3e(m,k,j,i) = emfz_(m,k,j,i)*mbsize.d_view(m).dx2;
  });
  Kokkos::deep_copy(DevExeSpace(), eoa.x2e, 0.0);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::SendEMF_OA()
//! \brief With mesh refinement: send the remap EMFs (E*dt) on MeshBlock boundaries,
//! restricted where the neighbour is coarser, as for the stage EMFs.

TaskStatus OrbitalAdvectionFC::SendEMF_OA() {
  if (!flux_correct) return TaskStatus::complete;
  return pbval_oa->PackAndSendFluxFC(eoa);
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::RecvEMFAndUpdateB_OA()
//! \brief With mesh refinement: receive the remap EMFs, average shared edges with
//! same-level neighbours and replace them by the restricted EMFs of finer neighbours
//! (MeshBoundaryValuesFC::RecvAndUnpackFluxFC), then update B with CT.

TaskStatus OrbitalAdvectionFC::RecvEMFAndUpdateB_OA(DvceFaceFld4D<Real> &b0) {
  if (!flux_correct) return TaskStatus::complete;
  TaskStatus tstat = pbval_oa->RecvAndUnpackFluxFC(eoa);
  if (tstat != TaskStatus::complete) return tstat;

  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &eoa_ = eoa;
  auto &emfx_ = emfx;
  auto &emfz_ = emfz;
  par_for("oa-emf1", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    emfx_(m,k,j,i) = eoa_.x1e(m,k,j,i)/mbsize.d_view(m).dx2;
  });
  par_for("oa-emf3", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    emfz_(m,k,j,i) = eoa_.x3e(m,k,j,i)/mbsize.d_view(m).dx2;
  });
  UpdateB(b0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::UpdateB()
//! \brief CT update of the face-centred field from the remap EMFs emfx, emfz.

void OrbitalAdvectionFC::UpdateB(DvceFaceFld4D<Real> &b0) {
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &emfx_ = emfx;
  auto &emfz_ = emfz;

  // Update face-centered fields using CT
  //---- update B1 (only for 2D/3D problems)
  if (pmy_pack->pmesh->multi_d) {
    par_for("oaCT-b1", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) -= (emfz_(m,k,j+1,i) - emfz_(m,k,j,i));
    });
  }

  //---- update B2 (curl terms in 1D and 3D problems)
  const bool &three_d_ = pmy_pack->pmesh->three_d;
  par_for("oaCT-b2", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dydx = mbsize.d_view(m).dx2/mbsize.d_view(m).dx1;
    b0.x2f(m,k,j,i) += dydx*(emfz_(m,k,j,i+1) - emfz_(m,k,j,i));
    if (three_d_) {
      Real dydz = mbsize.d_view(m).dx2/mbsize.d_view(m).dx3;
      b0.x2f(m,k,j,i) -= dydz*(emfx_(m,k+1,j,i) - emfx_(m,k,j,i));
    }
  });

  //---- update B3 (curl terms in 1D and 2D/3D problems)
  if (pmy_pack->pmesh->multi_d) {
    par_for("oaCT-b3", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x3f(m,k,j,i) += (emfx_(m,k,j+1,i) - emfx_(m,k,j,i));
    });
  }
}
