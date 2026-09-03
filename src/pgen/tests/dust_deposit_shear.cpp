//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_deposit_shear.cpp
//! \brief Unit test of the additive ghost-deposit exchange across shear-periodic x1
//! faces (dust track, Phase 4b).  No particles are involved: the x1 ghost slabs of the
//! MeshBlocks on the two shear faces are filled with known data, every other cell is
//! zero, and the exchange is run exactly as the dust module runs it (plain additive
//! exchange, then MeshBoundaryValuesDep::FoldShearDeposit).
//!
//! The fold is the adjoint of the shear-periodic ghost fill: an inner-ghost deposit at
//! (x, y) belongs to the outer boundary at (x + Lx, y - yshear), so the active edge
//! strips of the outer-face blocks must receive the inner-ghost content shifted to
//! y + yshear and those of the inner-face blocks the outer-ghost content shifted to
//! y - yshear, with yshear = q*Omega*Lx*time0.  Ghost deposits are PARTIAL sums: the
//! y/z ghost rows of a block's slab overlap the active rows of its y/z neighbors' slabs
//! and the exchange must ADD them.  Three fields separate the two properties:
//!   field 0  smooth g(x,y,z), periodic in y and z, on the slab rows within the block's
//!            own y/z range only (every physical location has exactly one source):
//!            received strips must equal g at the shifted location -> the remap
//!            interpolation error (zero for integer shifts, 2nd order for plm)
//!   field 1  constant 1 on the same rows: exact for any shift (constant rows remap
//!            exactly)
//!   field 2  constant 1 on the WHOLE slab incl. the y/z corner ghosts: the received
//!            value is the number of slab rows (of all face blocks) that overlap the
//!            source location, mult = (1 + #y-ghost overlaps)(1 + #z-ghost overlaps);
//!            checked against that integer expectation, exact at integer shifts (at a
//!            fractional shift the step profile carries an O(1) remap error at the
//!            block boundaries: reported, not a defect)
//! Printed ("# DEPOSIT-SHEAR ERRORS"): max_rel/l1_rel (field 0), const_err (field 1),
//! mult_err (field 2), leak = largest |value| of any field in an active cell outside
//! the receiving strips (must be exactly 0: catches the unsheared plain-periodic x1
//! contribution, which the fold must replace, and any mis-addressed row), and mass_err,
//! the relative change of the field-0 total (round-off: the remap is conservative).
//! Parameters: <problem> time0, remap (dc | plm | ppmx), fold (false = plain pass only:
//! then the strips must stay exactly 0).  Requires a 3D mesh with shear-periodic x1 and
//! a <shearing_box> block; run with <time> nlim = 0.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "bvals/bvals.hpp"
#include "pgen/pgen.hpp"

namespace {

constexpr Real TWOPI = 6.283185307179586;

// smooth, periodic in y and z; positive (>= 0.25) so relative errors are well defined
KOKKOS_INLINE_FUNCTION
Real TestFunction(Real x, Real y, Real z, Real lx, Real ly, Real lz) {
  return 1.0 + 0.5*sin(TWOPI*y/ly + 0.3)*cos(TWOPI*z/lz + 0.7)
             + 0.25*cos(2.0*TWOPI*y/ly)*(x/lx);
}

// number of slab rows (own + neighbors' ghost rows) overlapping global cell g of a
// periodic direction tiled by blocks of n cells with ng ghost rows
KOKKOS_INLINE_FUNCTION
int Multiplicity(int g, int gn, int n, int ng) {
  g = (g % gn + gn) % gn;
  int loc = g % n;
  int mult = 1;
  if (loc < ng) {mult += 1;}        // lower neighbor's upper ghost rows
  if (loc >= n - ng) {mult += 1;}   // upper neighbor's lower ghost rows
  return mult;
}

void Require(TaskStatus s, const char *what) {
  if (s == TaskStatus::fail) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "deposit exchange step failed: " << what << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::DustDepositShear()

void ProblemGenerator::DustDepositShear(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  Mesh *pm = pmy_mesh_;
  if (!(pm->three_d) ||
      (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dust_deposit_shear needs a 3D mesh with shear-periodic "
              << "x1 boundaries" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ncells1 = nx1 + 2*ng, ncells2 = nx2 + 2*ng, ncells3 = nx3 + 2*ng;
  const int nmb = pmbp->nmb_thispack;
  const int gny = pm->mesh_indcs.nx2, gnz = pm->mesh_indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &msize = pm->mesh_size;
  const Real lx = msize.x1max - msize.x1min;
  const Real ly = msize.x2max - msize.x2min;
  const Real lz = msize.x3max - msize.x3min;
  const Real dy = ly/static_cast<Real>(gny);

  // per-block global offsets of the first active cell (host -> device)
  DvceArray2D<int> goffs("dds_goffs", nmb, 2);
  {
    auto goffs_h = Kokkos::create_mirror_view(goffs);
    for (int m=0; m<nmb; ++m) {
      LogicalLocation &lloc = pm->lloc_eachmb[m + pmbp->gids];
      goffs_h(m,0) = static_cast<int>(lloc.lx2)*nx2;
      goffs_h(m,1) = static_cast<int>(lloc.lx3)*nx3;
    }
    Kokkos::deep_copy(goffs, goffs_h);
  }

  // gas at rest: the mesh carries a hydro module, nothing evolves (nlim = 0)
  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    const int nhydro = pmbp->phydro->nhydro;
    par_for("dds_gas", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2-1, 0, ncells1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int n = 0; n < nhydro; ++n) {u0(m,n,k,j,i) = 0.0;}
      u0(m,IDN,k,j,i) = 1.0;
      if (nhydro > 4) {u0(m,IEN,k,j,i) = 1.0;}
    });
  }

  // parameters
  const Real qshear = pin->GetReal("shearing_box","qshear");
  const Real omega0 = pin->GetReal("shearing_box","omega0");
  const Real time0 = pin->GetOrAddReal("problem","time0",0.37);
  const bool do_fold = pin->GetOrAddBoolean("problem","fold",true);
  std::string rmap = pin->GetOrAddString("problem","remap","plm");
  ReconstructionMethod rcon;
  if (rmap == "dc") {
    rcon = ReconstructionMethod::dc;
  } else if (rmap == "plm") {
    rcon = ReconstructionMethod::plm;
  } else if (rmap == "ppmx") {
    rcon = ReconstructionMethod::ppmx;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<problem>/remap must be dc, plm, or ppmx" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real ysh = qshear*omega0*lx*time0;
  // integer part of the shift in cells (for the multiplicity expectation of field 2)
  const int jshift = static_cast<int>(std::floor(ysh/dy + 0.5));
  const bool integer_shift = (std::fabs(ysh/dy - static_cast<Real>(jshift)) < 1.0e-10);

  // deposit field: (0) smooth on own rows, (1) constant on own rows, (2) constant on
  // the whole slab incl. corner ghosts
  const int nvar = 3;
  DvceArray5D<Real> a("deptest", nmb, nvar, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(a, 0.0);

  par_for("dds_fill", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    const Real dx1 = size.d_view(m).dx1;
    const Real y = size.d_view(m).x2min + (static_cast<Real>(j - js) + 0.5)*size.d_view(m).dx2;
    const Real z = size.d_view(m).x3min + (static_cast<Real>(k - ks) + 0.5)*size.d_view(m).dx3;
    const bool own = (j >= js) && (j <= je) && (k >= ks) && (k <= ke);
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      for (int i = 0; i < is; ++i) {
        const Real x = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*dx1;
        if (own) {
          a(m,0,k,j,i) = TestFunction(x, y, z, lx, ly, lz);
          a(m,1,k,j,i) = 1.0;
        }
        a(m,2,k,j,i) = 1.0;
      }
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      for (int i = ie+1; i < ncells1; ++i) {
        const Real x = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*dx1;
        if (own) {
          a(m,0,k,j,i) = TestFunction(x, y, z, lx, ly, lz);
          a(m,1,k,j,i) = 1.0;
        }
        a(m,2,k,j,i) = 1.0;
      }
    }
  });

  // total deposited mass (field 0) before the exchange: all ghost slabs
  const int ncell = ncells3*ncells2*ncells1;
  Real mass_before = 0.0;
  Kokkos::parallel_reduce("dds_mass0", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*ncell),
  KOKKOS_LAMBDA(const int idx, Real &sum) {
    const int m = idx/ncell;
    const int r = idx - m*ncell;
    const int k = r/(ncells2*ncells1);
    const int j = (r - k*ncells2*ncells1)/ncells1;
    const int i = r - k*ncells2*ncells1 - j*ncells1;
    sum += a(m,0,k,j,i);
  }, Kokkos::Sum<Real>(mass_before));

  // the exchange exactly as the dust module performs it
  MeshBoundaryValuesDep dep(pmbp, pin);
  dep.InitializeBuffers(nvar);
  Require(dep.InitRecv(nvar), "InitRecv");
  Require(dep.PackAndSendDeposit(a), "PackAndSendDeposit");
  TaskStatus st;
  do {
    st = dep.RecvAndSumDeposit(a);
    Require(st, "RecvAndSumDeposit");
  } while (st == TaskStatus::incomplete);
  Require(dep.ClearSend(), "ClearSend");
  Require(dep.ClearRecv(), "ClearRecv");
  if (do_fold) {
    Require(dep.FoldShearDeposit(a, ysh, rcon), "FoldShearDeposit");
  }

  // errors over the active cells
  const int nact = nx3*nx2*nx1;
  Real max_err = 0.0, l1_err = 0.0, l1_ref = 0.0, const_err = 0.0, mult_err = 0.0;
  Real leak = 0.0, mass_after = 0.0;
  Kokkos::parallel_reduce("dds_err", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nact),
  KOKKOS_LAMBDA(const int idx, Real &emax, Real &e1, Real &r1, Real &ec, Real &em,
                Real &lk, Real &ms) {
    const int m = idx/nact;
    const int r = idx - m*nact;
    const int k = r/(nx2*nx1) + ks;
    const int j = (r - (k-ks)*nx2*nx1)/nx1 + js;
    const int i = r - (k-ks)*nx2*nx1 - (j-js)*nx1 + is;
    const Real x = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*size.d_view(m).dx1;
    const Real y = size.d_view(m).x2min + (static_cast<Real>(j - js) + 0.5)*size.d_view(m).dx2;
    const Real z = size.d_view(m).x3min + (static_cast<Real>(k - ks) + 0.5)*size.d_view(m).dx3;
    const int gj = goffs(m,0) + (j - js);
    const int gk = goffs(m,1) + (k - ks);
    const bool outer = (mb_bcs.d_view(m, BoundaryFace::outer_x1) ==
                        BoundaryFlag::shear_periodic) && (i > ie - ng);
    const bool inner = (mb_bcs.d_view(m, BoundaryFace::inner_x1) ==
                        BoundaryFlag::shear_periodic) && (i < is + ng);
    Real expect = 0.0, expect_c = 0.0, expect_m = 0.0;
    if (do_fold) {
      if (outer) {   // inner-ghost content, shifted by -yshear: source row gj + jshift
        expect += TestFunction(x - lx, y + ysh, z, lx, ly, lz);
        expect_c += 1.0;
        expect_m += static_cast<Real>(Multiplicity(gj + jshift, gny, nx2, ng)
                                      *Multiplicity(gk, gnz, nx3, ng));
      }
      if (inner) {   // outer-ghost content, shifted by +yshear: source row gj - jshift
        expect += TestFunction(x + lx, y - ysh, z, lx, ly, lz);
        expect_c += 1.0;
        expect_m += static_cast<Real>(Multiplicity(gj - jshift, gny, nx2, ng)
                                      *Multiplicity(gk, gnz, nx3, ng));
      }
    }
    ms += a(m,0,k,j,i);
    if (outer || inner) {
      const Real d = fabs(a(m,0,k,j,i) - expect);
      emax = fmax(emax, d/fmax(fabs(expect), 1.0e-300));
      e1 += d;
      r1 += fabs(expect);
      ec = fmax(ec, fabs(a(m,1,k,j,i) - expect_c));
      em = fmax(em, fabs(a(m,2,k,j,i) - expect_m));
    } else {
      lk = fmax(lk, fmax(fabs(a(m,0,k,j,i)),
                         fmax(fabs(a(m,1,k,j,i)), fabs(a(m,2,k,j,i)))));
    }
  }, Kokkos::Max<Real>(max_err), Kokkos::Sum<Real>(l1_err), Kokkos::Sum<Real>(l1_ref),
     Kokkos::Max<Real>(const_err), Kokkos::Max<Real>(mult_err), Kokkos::Max<Real>(leak),
     Kokkos::Sum<Real>(mass_after));

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &max_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &const_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mult_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &leak, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &l1_err, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &l1_ref, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mass_before, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mass_after, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (global_variable::my_rank == 0) {
    std::printf("# DEPOSIT-SHEAR ERRORS: nx2= %d ng= %d remap= %s fold= %d time0= %.4f "
                "yshear/dy= %.6f (integer: %d) max_rel= %.6e l1_rel= %.6e "
                "const_err= %.3e mult_err= %.3e leak= %.3e mass_err= %.3e\n",
                gny, ng, rmap.c_str(), (do_fold ? 1 : 0), time0, ysh/dy,
                (integer_shift ? 1 : 0), max_err, (l1_ref > 0.0 ? l1_err/l1_ref : l1_err),
                const_err, mult_err, leak,
                std::fabs(mass_after - mass_before)/mass_before);
  }
  return;
}
