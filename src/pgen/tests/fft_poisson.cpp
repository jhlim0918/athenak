//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fft_poisson.cpp
//! \brief Problem generator to validate the FFT self-gravity solver. Initializes a
//! deterministic density field (multi-mode + Gaussian blob, or a horizontally uniform
//! isothermal slab with <problem> profile = slab), calls the gravity solve once, then
//! measures the discrete residual
//!     L[phi] - four_pi_G*(rho - rho_mean)      (periodic vertical BC)
//!     L[phi] - four_pi_G*rho                   (open vertical BC)
//! where L is the same 7-point Laplacian differenced by the SelfGravity source term,
//! evaluated with the ghost zones the solver filled. In a non-shearing periodic box the
//! residual is machine precision; with shear it converges at the remap order; with
//! vert_bc=open the layers touching the extrapolated x3 ghosts are excluded from the
//! "interior" norms. Set <problem> time0 > 0 to exercise a nonzero shear phase qomt.
//! For profile=slab the potential is also compared to the analytic infinite-slab
//! solution phi(z) = four_pi_G*rho0*H^2*ln(cosh((z-z0)/H)) (up to a constant).

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#if FFT_ENABLED
#include "gravity/fft_gravity.hpp"
#endif
#include "pgen/pgen.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::FFTPoisson()
//! \brief sets up density field, solves for Phi, and reports discrete residual norms

void ProblemGenerator::FFTPoisson(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pgrav == nullptr) {
    std::cout << "### FATAL ERROR in ProblemGenerator::FFTPoisson" << std::endl
              << "fft_poisson pgen requires a <gravity> block in the input file"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  bool use_mhd = (pmbp->pmhd != nullptr);
  std::string soe = use_mhd ? "mhd" : "hydro";

  // gravitational constant (same duplicated-setter pattern as other gravity pgens)
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  pmbp->pgrav->four_pi_G = four_pi_G;
  if (pmbp->pgrav->pmgd != nullptr) {
    pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);
  }

  if (restart) return;

  // problem parameters
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real amp = pin->GetOrAddReal("problem", "amp", 0.1);
  Real blob_amp = pin->GetOrAddReal("problem", "blob_amp", 1.0);
  Real time0 = pin->GetOrAddReal("problem", "time0", 0.0);
  std::string profile = pin->GetOrAddString("problem", "profile", "modes");
  bool slab = (profile == "slab");
  bool shwave = (profile == "shwave");
  // sheet: single horizontal Fourier mode (n1,n2) on ONE z-layer (sheet_k).  With
  // open/slab vertical BCs the exact solution of the DISCRETE Poisson equation on an
  // infinite vacuum stack is C*mu^|k-sheet_k| * (same mode), with no gauge freedom
  // for kperp != 0 -- an absolute (constant included) test of the mg_bc=slab planes.
  bool sheet = (profile == "sheet");
  // sin3: the Tomida & Stone (2023) sec. 4.1 triple-sine wave with an analytic
  // potential; enables the per-iteration convergence study (<problem> conv_niter)
  bool sin3 = (profile == "sin3");
  // shwave: single rolled-frame Fourier mode (a slanted wave in the current frame),
  // whose discrete solution is analytic: phi = -|amp*rho0*four_pi_G/D| * same wave
  int wn1 = pin->GetOrAddInteger("problem", "n1", 1);
  int wn2 = pin->GetOrAddInteger("problem", "n2", 1);
  int wn3 = pin->GetOrAddInteger("problem", "n3", 1);

  // setting the mesh time before the solve exercises a nonzero shear phase (qomt)
  if (time0 > 0.0) pmy_mesh_->time = time0;

  std::string eos_type = pin->GetString(soe, "eos");
  bool is_ideal = !(eos_type == "isothermal");
  Real gm1 = 0.0, p0 = 0.0;
  if (is_ideal) {
    Real gamma = pin->GetOrAddReal(soe, "gamma", 5.0/3.0);
    gm1 = gamma - 1.0;
    p0 = pin->GetOrAddReal("problem", "p0", 1.0);
  }

  // domain geometry
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min;
  Real ly = msize.x2max - msize.x2min;
  Real lz = msize.x3max - msize.x3min;
  Real xc = 0.5*(msize.x1min + msize.x1max);
  Real yc = 0.5*(msize.x2min + msize.x2max);
  Real zc = 0.5*(msize.x3min + msize.x3max);
  Real slab_h = pin->GetOrAddReal("problem", "slab_h", 0.1*lz);
  Real blob_w = 0.1*lx;

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = use_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;

  int nz_mesh = pmy_mesh_->mesh_indcs.nx3;
  Real dz_mesh = lz/static_cast<Real>(nz_mesh);
  int sheet_k = pin->GetOrAddInteger("problem", "sheet_k", nz_mesh/2);
  Real dom_zmin = msize.x3min;

  // shear phase at the solve time (same formula as FFTGravitySolver::ComputeQomt)
  Real qomt = 0.0;
  if (pin->DoesBlockExist("shearing_box")) {
    Real qsh = pin->GetReal("shearing_box", "qshear");
    Real om0 = pin->GetReal("shearing_box", "omega0");
    if (qsh != 0.0 && om0 != 0.0) {
      Real tshear = ly/(qsh*om0*lx);
      Real dts = time0 - std::floor(time0/tshear)*tshear;
      qomt = qsh*om0*dts;
    }
  }

  // initialize density (deterministic), zero velocities
  par_for("fftp_init", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
    Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    Real rho;
    if (slab) {
      Real sech = 1.0/cosh((z - zc)/slab_h);
      rho = rho0*sech*sech + 1.0e-10*rho0;
    } else if (sin3) {
      rho = rho0 + amp*sin(2.0*M_PI*x/lx)*sin(2.0*M_PI*y/ly)*sin(2.0*M_PI*z/lz);
    } else if (shwave) {
      // single rolled-frame mode, slanted into the current frame by the shear phase
      Real arg = 2.0*M_PI*(wn1*x/lx + wn2*(y + qomt*x)/ly + wn3*z/lz);
      rho = rho0*(1.0 + amp*cos(arg));
    } else if (sheet) {
      // uniform background + single rolled-frame horizontal mode on one z-layer
      int kg = static_cast<int>((z - dom_zmin)/dz_mesh);
      Real arg = 2.0*M_PI*(wn1*x/lx + wn2*(y + qomt*x)/ly);
      rho = rho0 + ((kg == sheet_k) ? amp*cos(arg) : 0.0);
    } else {
      Real r2 = SQR(x - xc) + SQR(y - yc) + SQR(z - zc);
      rho = rho0*(1.0
          + amp*sin(2.0*2.0*M_PI*(x - xc)/lx)*cos(3.0*2.0*M_PI*(y - yc)/ly)
               *cos(1.0*2.0*M_PI*(z - zc)/lz)
          + amp*cos(1.0*2.0*M_PI*(x - xc)/lx)*sin(2.0*2.0*M_PI*(y - yc)/ly)
               *sin(2.0*2.0*M_PI*(z - zc)/lz))
          + blob_amp*rho0*exp(-r2/SQR(blob_w));
    }
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    if (is_ideal) u0(m,IEN,k,j,i) = p0/gm1;
  });

  // ---- dust twin (dust track, Phase 4c) ----------------------------------------------
  // With <problem> dust_frac = f > 0, a fraction f of the density profile is carried by
  // a lattice of dust particles (one per cell, at the cell centres, mass f*rho*V) and the
  // gas keeps (1-f)*rho.  With <dust> deposit = ngp the particle-mesh density is the
  // profile to round-off and every metric below must match the gas-only run; with tsc
  // the dust part is the TSC-filtered profile.  The dust module assembles its density
  // synchronously here (the solve is called outside the time loop).
  Real dust_frac = pin->GetOrAddReal("problem", "dust_frac", 0.0);
  bool dust_twin = (dust_frac > 0.0);
  // dust_part = all (default): the dust carries the fraction f of the whole profile;
  // blob (profile = modes only): the dust carries f times the Gaussian blob and the gas
  // everything else, so that the two components have DIFFERENT shapes and exert a
  // nonzero net force on each other -- the momentum-antisymmetry check below
  std::string dust_part = pin->GetOrAddString("problem", "dust_part", "all");
  bool dust_blob = (dust_part == "blob");
  if (dust_blob && (slab || sin3 || shwave || sheet)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<problem>/dust_part = blob needs profile = modes"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (dust_twin) {
    if (pmbp->pdust == nullptr || pmbp->ppart == nullptr || !(pmbp->pdust->gravity)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<problem>/dust_frac > 0 requires <particles> and <dust> "
                << "blocks with <dust>/gravity = true" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    particles::Particles *ppar = pmbp->ppart;
    int npart = ppar->nprtcl_thispack;
    int lnx1 = indcs.nx1, lnx2 = indcs.nx2, lnx3 = indcs.nx3;
    int ncells = lnx1*lnx2*lnx3;
    if (npart != nmb*ncells) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "the dust twin requires <particles>/ppc = 1 (one "
                << "particle per cell)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto &pr = ppar->prtcl_rdata;
    auto &pi = ppar->prtcl_idata;
    auto gids = pmbp->gids;
    auto &taus_ = pmbp->pdust->taus;
    Real f = dust_frac;
    par_for("fftp_dust", DevExeSpace(), 0, (npart-1), KOKKOS_LAMBDA(const int p) {
      int m = p/ncells;
      int c = p - m*ncells;
      int i = c % lnx1;
      int j = (c/lnx1) % lnx2;
      int k = c/(lnx1*lnx2);
      pr(IPX,p) = CellCenterX(i, lnx1, size.d_view(m).x1min, size.d_view(m).x1max);
      pr(IPY,p) = CellCenterX(j, lnx2, size.d_view(m).x2min, size.d_view(m).x2max);
      pr(IPZ,p) = CellCenterX(k, lnx3, size.d_view(m).x3min, size.d_view(m).x3max);
      pi(PGID,p) = gids + m;
      pi(PSP,p) = 0;
      pr(IPTS,p) = taus_.d_view(0);
      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      if (dust_blob) {
        Real r2 = SQR(pr(IPX,p) - xc) + SQR(pr(IPY,p) - yc) + SQR(pr(IPZ,p) - zc);
        pr(IPM,p) = f*blob_amp*rho0*exp(-r2/SQR(blob_w))*vol;
      } else {
        pr(IPM,p) = f*u0(m,IDN,ks+k,js+j,is+i)*vol;
      }
      pr(IPVX,p) = 0.0; pr(IPVY,p) = 0.0; pr(IPVZ,p) = 0.0;
      pr(IPRX,p) = 0.0; pr(IPRY,p) = 0.0; pr(IPRZ,p) = 0.0;
    });
    Real rfloor = pin->GetOrAddReal("problem", "dust_gas_floor", 1.0e-10)*rho0;
    par_for("fftp_gas_scale", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (dust_blob) {
        Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        Real r2 = SQR(x - xc) + SQR(y - yc) + SQR(z - zc);
        Real blob = f*blob_amp*rho0*exp(-r2/SQR(blob_w));
        u0(m,IDN,k,j,i) = fmax(u0(m,IDN,k,j,i) - blob, rfloor);
      } else {
        u0(m,IDN,k,j,i) = fmax((1.0 - f)*u0(m,IDN,k,j,i), rfloor);
      }
    });
    pmbp->pdust->AssembleDustDensityNow();
  }
  // the Poisson source as the solver sees it (gas, or gas + dust once registered)
  auto src = pmbp->pgrav->SourceArray();
  const int isrc = pmbp->pgrav->SourceIndex();

  // ---- per-iteration convergence study (Tomida & Stone 2023, sec. 4.1) ---------------
  // With <problem> conv_niter = N and solver=multigrid, run N successive V-cycle
  // iterations (in MGI mode each Solve() warm-starts from pgrav->phi, so N calls with
  // niteration=1 are N V-cycles; in FMG mode iteration 1 is one full FMG sweep and the
  // rest are V-cycles -- the conventions of their Figure 5). After each iteration print
  //   eps    = rms(phi - phi_analytic)           (their eq. 6, means subtracted)
  //   delta  = rms(phi - phi_fully_converged)    (their eq. 5; reference = the solution
  //            after conv_niter iterations, built by a bit-identical first pass)
  //   defect = rms(4piG*(rho-rho_mean) - L[phi]) (their eq. 7)
  int conv_niter = pin->GetOrAddInteger("problem", "conv_niter", 0);
  bool conv_study = (conv_niter > 0) && (pmbp->pgrav->pmgd != nullptr) && sin3;
  if (conv_study) {
    auto &phi = pmbp->pgrav->phi;
    auto pmgd = pmbp->pgrav->pmgd;
    bool fmg_mode = pin->GetOrAddBoolean("gravity", "full_multigrid", false);

    int ni = ie - is + 1, nj = je - js + 1, nk = ke - ks + 1;
    int nmkji = nmb*nk*nj*ni;
    Real ncells_tot = static_cast<Real>(pmy_mesh_->mesh_indcs.nx1)
                     *static_cast<Real>(pmy_mesh_->mesh_indcs.nx2)
                     *static_cast<Real>(pmy_mesh_->mesh_indcs.nx3);
    Real phi_amp3 = -four_pi_G*amp/(SQR(2.0*M_PI/lx) + SQR(2.0*M_PI/ly)
                                    + SQR(2.0*M_PI/lz));

    // mean density for the defect RHS
    Real rsum = 0.0;
    Kokkos::parallel_reduce("ts41_mean", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &lsum) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      lsum += src(m,isrc,k,j,i);
    }, Kokkos::Sum<Real>(rsum));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &rsum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    Real rmean = rsum/ncells_tot;

    DvceArray5D<Real> phi_ref("ts41_ref", phi.extent(0), phi.extent(1),
                              phi.extent(2), phi.extent(3), phi.extent(4));

    // metric evaluators (each returns a volume-weighted rms over active zones)
    Real lx_c = lx, ly_c = ly, lz_c = lz;
    auto rms_eps = [&]() -> Real {
      Real snum = 0.0, sana = 0.0;
      Kokkos::parallel_reduce("ts41_means",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, Real &ln, Real &la) {
        int i = is + (idx % ni);
        int j = js + ((idx/ni) % nj);
        int k = ks + ((idx/(ni*nj)) % nk);
        int m = idx/(ni*nj*nk);
        Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
        Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
        Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
        Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
        Real y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
        ln += phi(m,0,k,j,i);
        la += phi_amp3*sin(2.0*M_PI*x/lx_c)*sin(2.0*M_PI*y/ly_c)*sin(2.0*M_PI*z/lz_c);
      }, Kokkos::Sum<Real>(snum), Kokkos::Sum<Real>(sana));
#if MPI_PARALLEL_ENABLED
      Real ms[2] = {snum, sana};
      MPI_Allreduce(MPI_IN_PLACE, ms, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      snum = ms[0]; sana = ms[1];
#endif
      Real mnum = snum/ncells_tot, mana = sana/ncells_tot;
      Real ssq = 0.0;
      Kokkos::parallel_reduce("ts41_eps",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, Real &lsq) {
        int i = is + (idx % ni);
        int j = js + ((idx/ni) % nj);
        int k = ks + ((idx/(ni*nj)) % nk);
        int m = idx/(ni*nj*nk);
        Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
        Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
        Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
        Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
        Real y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
        Real ana = phi_amp3*sin(2.0*M_PI*x/lx_c)*sin(2.0*M_PI*y/ly_c)
                          *sin(2.0*M_PI*z/lz_c);
        lsq += SQR((phi(m,0,k,j,i) - mnum) - (ana - mana));
      }, Kokkos::Sum<Real>(ssq));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &ssq, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
      return std::sqrt(ssq/ncells_tot);
    };
    auto rms_delta = [&]() -> Real {
      Real ssq = 0.0;
      Kokkos::parallel_reduce("ts41_delta",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, Real &lsq) {
        int i = is + (idx % ni);
        int j = js + ((idx/ni) % nj);
        int k = ks + ((idx/(ni*nj)) % nk);
        int m = idx/(ni*nj*nk);
        lsq += SQR(phi(m,0,k,j,i) - phi_ref(m,0,k,j,i));
      }, Kokkos::Sum<Real>(ssq));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &ssq, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
      return std::sqrt(ssq/ncells_tot);
    };
    auto rms_defect = [&]() -> Real {
      Real ssq = 0.0;
      Kokkos::parallel_reduce("ts41_defect",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, Real &lsq) {
        int i = is + (idx % ni);
        int j = js + ((idx/ni) % nj);
        int k = ks + ((idx/(ni*nj)) % nk);
        int m = idx/(ni*nj*nk);
        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;
        Real lap = (phi(m,0,k,j,i+1) - 2.0*phi(m,0,k,j,i) + phi(m,0,k,j,i-1))/SQR(dx1)
                 + (phi(m,0,k,j+1,i) - 2.0*phi(m,0,k,j,i) + phi(m,0,k,j-1,i))/SQR(dx2)
                 + (phi(m,0,k+1,j,i) - 2.0*phi(m,0,k,j,i) + phi(m,0,k-1,j,i))/SQR(dx3);
        lsq += SQR(four_pi_G*(src(m,isrc,k,j,i) - rmean) - lap);
      }, Kokkos::Sum<Real>(ssq));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &ssq, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
      return std::sqrt(ssq/ncells_tot);
    };
    auto report = [&](int it) {
      Real e = rms_eps(), dl = rms_delta(), df = rms_defect();
      if (global_variable::my_rank == 0) {
        std::cout << std::scientific << std::setprecision(6)
                  << "# TS41: iter= " << it << " eps= " << e
                  << " delta= " << dl << " defect= " << df << std::endl;
      }
    };

    // pass 0 builds the fully-converged reference; pass 1 repeats bit-identically
    // (cycle parity reset) and records the metrics, so delta(conv_niter) == 0
    for (int pass = 0; pass < 2; ++pass) {
      bool record = (pass == 1);
      Kokkos::deep_copy(phi, 0.0);
      pmgd->ResetCycleParity();
      pmgd->SetFullMultigrid(fmg_mode);
      pmgd->SetNumIterations(fmg_mode ? 0 : 1);
      if (record && !fmg_mode) report(0);  // MGI: the (zero) initial guess
      for (int it = 1; it <= conv_niter; ++it) {
        pmbp->pgrav->Solve(nullptr, 1);
        if (fmg_mode && it == 1) {  // after the FMG sweep, continue with V-cycles
          pmgd->SetFullMultigrid(false);
          pmgd->SetNumIterations(1);
        }
        if (record) report(it);
      }
      if (!record) Kokkos::deep_copy(phi_ref, phi);
    }
  } else {
    // solve for the potential with the configured solver
    pmbp->pgrav->Solve(nullptr, 1);
  }

  // ---- residual diagnostics -----------------------------------------------------------
  // mg_bc=slab (multigrid) and vert_bc=open (fft) are the same physical configuration
  bool mg_slab = (pin->GetOrAddString("gravity", "mg_bc", "none") == "slab");
  bool open_z = (pin->GetOrAddString("gravity", "vert_bc", "periodic") == "open")
                || mg_slab;
  bool shear = pin->DoesBlockExist("shearing_box");

  int ni = ie - is + 1, nj = je - js + 1, nk = ke - ks + 1;
  int nmkji = nmb*nk*nj*ni;
  Real ncells_tot = static_cast<Real>(pmy_mesh_->mesh_indcs.nx1)
                   *static_cast<Real>(pmy_mesh_->mesh_indcs.nx2)
                   *static_cast<Real>(pmy_mesh_->mesh_indcs.nx3);

  // mean density (subtracted from the RHS in the triply-periodic solve)
  Real rho_sum = 0.0;
  Kokkos::parallel_reduce("fftp_mean", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(int idx, Real &lsum) {
    int i = is + (idx % ni);
    int j = js + ((idx/ni) % nj);
    int k = ks + ((idx/(ni*nj)) % nk);
    int m = idx/(ni*nj*nk);
    lsum += src(m,isrc,k,j,i);
  }, Kokkos::Sum<Real>(rho_sum));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &rho_sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  Real rho_mean = open_z ? 0.0 : rho_sum/ncells_tot;

  auto &phi = pmbp->pgrav->phi;
  Real dom_x1min = msize.x1min, dom_x1max = msize.x1max;
  Real dom_x3min = msize.x3min, dom_x3max = msize.x3max;

  Real res_max = 0.0, res_sq = 0.0, rhs_max = 0.0;
  Real res_max_int = 0.0, res_sq_int = 0.0;
  Real nint_sum = 0.0;
  Kokkos::parallel_reduce("fftp_resid", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(int idx, Real &lmax, Real &lsq, Real &lrhs,
                Real &lmax_i, Real &lsq_i, Real &lnint) {
    int i = is + (idx % ni);
    int j = js + ((idx/ni) % nj);
    int k = ks + ((idx/(ni*nj)) % nk);
    int m = idx/(ni*nj*nk);
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;
    Real lap = (phi(m,0,k,j,i+1) - 2.0*phi(m,0,k,j,i) + phi(m,0,k,j,i-1))/SQR(dx1)
             + (phi(m,0,k,j+1,i) - 2.0*phi(m,0,k,j,i) + phi(m,0,k,j-1,i))/SQR(dx2)
             + (phi(m,0,k+1,j,i) - 2.0*phi(m,0,k,j,i) + phi(m,0,k-1,j,i))/SQR(dx3);
    Real rhs = four_pi_G*(src(m,isrc,k,j,i) - rho_mean);
    Real res = fabs(lap - rhs);
    lmax = fmax(lmax, res);
    lsq += SQR(res);
    lrhs = fmax(lrhs, fabs(rhs));
    // interior norms: exclude cells whose stencil touches approximate ghosts --
    // the x1 boundary layers when shearing (remap-interpolated shear-periodic ghosts)
    // and the x3 boundary layers when vert_bc=open (extrapolated ghosts)
    Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
    Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
    Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    bool edge_x1 = shear && ((x < dom_x1min + dx1) || (x > dom_x1max - dx1));
    bool edge_x3 = open_z && ((z < dom_x3min + dx3) || (z > dom_x3max - dx3));
    if (!edge_x1 && !edge_x3) {
      lmax_i = fmax(lmax_i, res);
      lsq_i += SQR(res);
      lnint += 1.0;
    }
  }, Kokkos::Max<Real>(res_max), Kokkos::Sum<Real>(res_sq), Kokkos::Max<Real>(rhs_max),
     Kokkos::Max<Real>(res_max_int), Kokkos::Sum<Real>(res_sq_int),
     Kokkos::Sum<Real>(nint_sum));
#if MPI_PARALLEL_ENABLED
  {
    Real maxes[3] = {res_max, rhs_max, res_max_int};
    Real sums[3] = {res_sq, res_sq_int, nint_sum};
    MPI_Allreduce(MPI_IN_PLACE, maxes, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    res_max = maxes[0]; rhs_max = maxes[1]; res_max_int = maxes[2];
    res_sq = sums[0]; res_sq_int = sums[1]; nint_sum = sums[2];
  }
#endif

  Real l2 = std::sqrt(res_sq/ncells_tot)/rhs_max;
  Real l2_int = (nint_sum > 0.0) ? std::sqrt(res_sq_int/nint_sum)/rhs_max : 0.0;
  if (global_variable::my_rank == 0) {
    std::cout << std::scientific << std::setprecision(6)
              << "# FFT-POISSON ERRORS: max_rel_all= " << res_max/rhs_max
              << " l2_rel_all= " << l2
              << " max_rel_int= " << res_max_int/rhs_max
              << " l2_rel_int= " << l2_int << std::endl;
  }

  // ---- analytic triple-sine comparison (profile = sin3, Tomida & Stone 4.1) -----------
  // rms of phi - phi_analytic (their eq. 6, means subtracted), for any solver
  if (sin3) {
    Real phi_amp3 = -four_pi_G*amp/(SQR(2.0*M_PI/lx) + SQR(2.0*M_PI/ly)
                                    + SQR(2.0*M_PI/lz));
    Real lx_c = lx, ly_c = ly, lz_c = lz;
    Real snum = 0.0, sana = 0.0;
    Kokkos::parallel_reduce("ts41f_means",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &ln, Real &la) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
      Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      ln += phi(m,0,k,j,i);
      la += phi_amp3*sin(2.0*M_PI*x/lx_c)*sin(2.0*M_PI*y/ly_c)*sin(2.0*M_PI*z/lz_c);
    }, Kokkos::Sum<Real>(snum), Kokkos::Sum<Real>(sana));
#if MPI_PARALLEL_ENABLED
    {
      Real ms[2] = {snum, sana};
      MPI_Allreduce(MPI_IN_PLACE, ms, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      snum = ms[0]; sana = ms[1];
    }
#endif
    Real ssq = 0.0;
    Kokkos::parallel_reduce("ts41f_eps",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &lsq) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
      Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real ana = phi_amp3*sin(2.0*M_PI*x/lx_c)*sin(2.0*M_PI*y/ly_c)
                        *sin(2.0*M_PI*z/lz_c);
      lsq += SQR((phi(m,0,k,j,i) - snum/ncells_tot) - (ana - sana/ncells_tot));
    }, Kokkos::Sum<Real>(ssq));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &ssq, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "# TS41-EPS: rms_eps= " << std::sqrt(ssq/ncells_tot) << std::endl;
    }
  }

  // ---- dust force check (Phase 4c): gather -grad(phi) at the particles ---------------
  // For profile = sin3 the analytic acceleration is known; the error of the gathered
  // force is the centred-difference truncation (NGP at cell centres: exactly the cell
  // value) plus the deposit/gather filtering (TSC).
  if (dust_twin && pmbp->pdust->gravity_force) {
    pmbp->pdust->ComputeGravForceNow();
    // Net force on the gas, sum_cells rho_g g V with the same centred gradient the gas
    // source term uses, and on the dust, sum_p m_p g(x_p) with the PM gather: their sum
    // is the momentum error of the particle-mesh coupling (zero in the continuum).
    {
      particles::Particles *ppar = pmbp->ppart;
      int npart = ppar->nprtcl_thispack;
      auto &pr = ppar->prtcl_rdata;
      auto &pi = ppar->prtcl_idata;
      auto gids = pmbp->gids;
      auto &gforce = pmbp->pdust->gforce;
      int scheme = static_cast<int>(pmbp->pdust->deposit);
      Real fg[3] = {0.0, 0.0, 0.0}, fd[3] = {0.0, 0.0, 0.0}, fdabs = 0.0;
      Kokkos::parallel_reduce("fftp_fgas",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, Real &fx, Real &fy, Real &fz) {
        int i = is + (idx % ni);
        int j = js + ((idx/ni) % nj);
        int k = ks + ((idx/(ni*nj)) % nk);
        int m = idx/(ni*nj*nk);
        Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        Real rv = u0(m,IDN,k,j,i)*vol;
        fx += rv*gforce(m,0,k,j,i);
        fy += rv*gforce(m,1,k,j,i);
        fz += rv*gforce(m,2,k,j,i);
      }, Kokkos::Sum<Real>(fg[0]), Kokkos::Sum<Real>(fg[1]), Kokkos::Sum<Real>(fg[2]));
      Kokkos::parallel_reduce("fftp_fdust",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &fx, Real &fy, Real &fz, Real &fa) {
        int m = pi(PGID,p) - gids;
        int ip, jp, kp;
        Real wx[3], wy[3], wz[3];
        dust::PMWeights(pr(IPX,p), size.d_view(m).x1min, size.d_view(m).x1max, indcs.nx1,
                        is, scheme, ip, wx);
        dust::PMWeights(pr(IPY,p), size.d_view(m).x2min, size.d_view(m).x2max, indcs.nx2,
                        js, scheme, jp, wy);
        dust::PMWeights(pr(IPZ,p), size.d_view(m).x3min, size.d_view(m).x3max, indcs.nx3,
                        ks, scheme, kp, wz);
        Real gx = 0.0, gy = 0.0, gz = 0.0;
        for (int c=0; c<3; ++c) {
          for (int b=0; b<3; ++b) {
            Real wcb = wz[c]*wy[b];
            if (wcb == 0.0) continue;
            for (int a=0; a<3; ++a) {
              Real w = wcb*wx[a];
              gx += w*gforce(m,0,kp+c-1,jp+b-1,ip+a-1);
              gy += w*gforce(m,1,kp+c-1,jp+b-1,ip+a-1);
              gz += w*gforce(m,2,kp+c-1,jp+b-1,ip+a-1);
            }
          }
        }
        Real mp = pr(IPM,p);
        fx += mp*gx; fy += mp*gy; fz += mp*gz;
        fa += mp*sqrt(gx*gx + gy*gy + gz*gz);
      }, Kokkos::Sum<Real>(fd[0]), Kokkos::Sum<Real>(fd[1]), Kokkos::Sum<Real>(fd[2]),
         Kokkos::Sum<Real>(fdabs));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, fg, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, fd, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &fdabs, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
      Real fsum = std::sqrt(SQR(fg[0]+fd[0]) + SQR(fg[1]+fd[1]) + SQR(fg[2]+fd[2]));
      Real fdmag = std::sqrt(SQR(fd[0]) + SQR(fd[1]) + SQR(fd[2]));
      if (global_variable::my_rank == 0) {
        std::cout << "# DUST-GRAVITY MOMENTUM: F_dust= (" << fd[0] << " " << fd[1] << " "
                  << fd[2] << ") F_gas= (" << fg[0] << " " << fg[1] << " " << fg[2]
                  << ") |F_gas+F_dust|/|F_dust|= " << fsum/fdmag
                  << " |F_gas+F_dust|/sum_p m|g|= " << fsum/fdabs << std::endl;
      }
    }
    if (sin3) {
      particles::Particles *ppar = pmbp->ppart;
      int npart = ppar->nprtcl_thispack;
      auto &pr = ppar->prtcl_rdata;
      auto &pi = ppar->prtcl_idata;
      auto gids = pmbp->gids;
      auto &gforce = pmbp->pdust->gforce;
      int scheme = static_cast<int>(pmbp->pdust->deposit);
      Real phi_amp3 = -four_pi_G*amp/(SQR(2.0*M_PI/lx) + SQR(2.0*M_PI/ly)
                                      + SQR(2.0*M_PI/lz));
      Real kx = 2.0*M_PI/lx, ky = 2.0*M_PI/ly, kz = 2.0*M_PI/lz;
      Real gmax = fabs(phi_amp3)*fmax(kx, fmax(ky, kz));
      Real emax = 0.0;
      Kokkos::parallel_reduce("fftp_dust_force",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &lmax) {
        int m = pi(PGID,p) - gids;
        int ip, jp, kp;
        Real wx[3], wy[3], wz[3];
        Real x = pr(IPX,p), y = pr(IPY,p), z = pr(IPZ,p);
        dust::PMWeights(x, size.d_view(m).x1min, size.d_view(m).x1max, indcs.nx1, is,
                        scheme, ip, wx);
        dust::PMWeights(y, size.d_view(m).x2min, size.d_view(m).x2max, indcs.nx2, js,
                        scheme, jp, wy);
        dust::PMWeights(z, size.d_view(m).x3min, size.d_view(m).x3max, indcs.nx3, ks,
                        scheme, kp, wz);
        Real gx = 0.0, gy = 0.0, gz = 0.0;
        for (int c=0; c<3; ++c) {
          for (int b=0; b<3; ++b) {
            Real wcb = wz[c]*wy[b];
            if (wcb == 0.0) continue;
            for (int a=0; a<3; ++a) {
              Real w = wcb*wx[a];
              gx += w*gforce(m,0,kp+c-1,jp+b-1,ip+a-1);
              gy += w*gforce(m,1,kp+c-1,jp+b-1,ip+a-1);
              gz += w*gforce(m,2,kp+c-1,jp+b-1,ip+a-1);
            }
          }
        }
        Real ax = -phi_amp3*kx*cos(kx*x)*sin(ky*y)*sin(kz*z);
        Real ay = -phi_amp3*ky*sin(kx*x)*cos(ky*y)*sin(kz*z);
        Real az = -phi_amp3*kz*sin(kx*x)*sin(ky*y)*cos(kz*z);
        Real e = fmax(fabs(gx - ax), fmax(fabs(gy - ay), fabs(gz - az)));
        lmax = fmax(lmax, e);
      }, Kokkos::Max<Real>(emax));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &emax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
      if (global_variable::my_rank == 0) {
        std::cout << "# DUST-GRAVITY FORCE ERROR: max_rel= " << emax/gmax << std::endl;
      }
    }
  }

  // ---- analytic shearing-wave comparison (profile = shwave) ----------------------------
  // The initialized density is a single Fourier mode of the rolled frame, so the exact
  // solution of the discrete (rolled) Poisson equation is the same mode with amplitude
  // amp*rho0*four_pi_G/D, evaluated with the solver's shear-shifted kernel. The
  // pointwise phi error then measures the full pipeline (roll -> solve -> unroll) and
  // converges at the remap interpolation order.
  if (shwave) {
    Real dx1m = lx/static_cast<Real>(pmy_mesh_->mesh_indcs.nx1);
    Real dx2m = ly/static_cast<Real>(pmy_mesh_->mesh_indcs.nx2);
    Real dx3m = lz/static_cast<Real>(pmy_mesh_->mesh_indcs.nx3);
    Real sfac = qomt*lx/ly;
    Real kxdx = (wn1 + sfac*wn2)*2.0*M_PI/pmy_mesh_->mesh_indcs.nx1;
    Real kydy = static_cast<Real>(wn2)*2.0*M_PI/pmy_mesh_->mesh_indcs.nx2;
    Real kzdz = static_cast<Real>(wn3)*2.0*M_PI/pmy_mesh_->mesh_indcs.nx3;
    Real dd = (2.0*std::cos(kxdx) - 2.0)/SQR(dx1m)
            + (2.0*std::cos(kydy) - 2.0)/SQR(dx2m)
            + (2.0*std::cos(kzdz) - 2.0)/SQR(dx3m);
    Real phi_amp = amp*rho0*four_pi_G/dd;
    Real lx_c = lx, ly_c = ly, lz_c = lz, qomt_c = qomt;
    int n1_c = wn1, n2_c = wn2, n3_c = wn3;

    Real sum_num = 0.0, sum_ana = 0.0;
    Kokkos::parallel_reduce("fftp_shw_mean",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &lnum, Real &lana) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
      Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real arg = 2.0*M_PI*(n1_c*x/lx_c + n2_c*(y + qomt_c*x)/ly_c + n3_c*z/lz_c);
      lnum += phi(m,0,k,j,i);
      lana += phi_amp*cos(arg);
    }, Kokkos::Sum<Real>(sum_num), Kokkos::Sum<Real>(sum_ana));
#if MPI_PARALLEL_ENABLED
    {
      Real sums[2] = {sum_num, sum_ana};
      MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      sum_num = sums[0]; sum_ana = sums[1];
    }
#endif
    Real mean_num = sum_num/ncells_tot;
    Real mean_ana = sum_ana/ncells_tot;

    Real dmax = 0.0, dsq = 0.0;
    Kokkos::parallel_reduce("fftp_shw_err",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &lmax, Real &lsq) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
      Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real arg = 2.0*M_PI*(n1_c*x/lx_c + n2_c*(y + qomt_c*x)/ly_c + n3_c*z/lz_c);
      Real diff = (phi(m,0,k,j,i) - mean_num) - (phi_amp*cos(arg) - mean_ana);
      lmax = fmax(lmax, fabs(diff));
      lsq += SQR(diff);
    }, Kokkos::Max<Real>(dmax), Kokkos::Sum<Real>(dsq));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &dsq, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

    if (global_variable::my_rank == 0) {
      std::cout << "# FFT-POISSON SHWAVE ERROR: max_rel= " << dmax/std::abs(phi_amp)
                << " l2_rel= " << std::sqrt(dsq/ncells_tot)/std::abs(phi_amp)
                << std::endl;
    }
  }

  // ---- analytic sheet comparison (profile = sheet, open/slab vertical BC) -------------
  // The discrete infinite-vacuum solution is known in closed form:
  //   phi = phi_bg(kg) + C * mu^|kg - sheet_k| * cos(arg)
  // with C = 4piG*amp*(-dz^2/sqrt(alpha^2-4)) for the sheet mode and, for the uniform
  // background (kperp = 0, symmetric slab gauge G(dn) = dz^2*|dn|/2),
  //   phi_bg(kg) = 4piG*rho0*(dz^2/2)*[kg(kg+1)/2 + (nz-1-kg)(nz-kg)/2].
  // NO mean subtraction: both sectors are gauge-fixed by the slab boundary planes, so
  // this tests the mg_bc=slab infrastructure absolutely (constant included).
  if (sheet && open_z) {
    Real dx1m = lx/static_cast<Real>(pmy_mesh_->mesh_indcs.nx1);
    Real dx2m = ly/static_cast<Real>(pmy_mesh_->mesh_indcs.nx2);
    Real sfac = qomt*lx/ly;
    Real kxdx = (wn1 + sfac*wn2)*2.0*M_PI/pmy_mesh_->mesh_indcs.nx1;
    Real kydy = static_cast<Real>(wn2)*2.0*M_PI/pmy_mesh_->mesh_indcs.nx2;
    Real lam = (2.0*std::cos(kxdx) - 2.0)/SQR(dx1m)
             + (2.0*std::cos(kydy) - 2.0)/SQR(dx2m);
    Real alpha = 2.0 - lam*SQR(dz_mesh);
    Real ssr = std::sqrt(alpha*alpha - 4.0);
    Real mu = 2.0/(alpha + ssr);
    Real camp = -four_pi_G*amp*SQR(dz_mesh)/ssr;
    Real bgfac = 0.5*four_pi_G*rho0*SQR(dz_mesh);
    Real lx_c = lx, ly_c = ly, qomt_c = qomt, dzm_c = dz_mesh, dzmin_c = dom_zmin;
    int n1_c = wn1, n2_c = wn2, k0_c = sheet_k, nz_c = nz_mesh;

    Real dmax = 0.0, dsq = 0.0;
    Kokkos::parallel_reduce("fftp_sheet_err",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &lmax, Real &lsq) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min, &x3max = size.d_view(m).x3max;
      Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      // real-valued root-layer coordinate: on the root grid cell centers sample
      // integer kgr exactly; on refined cells this evaluates the root-grid closed
      // form at the fine centers (correct to the interface discretization order,
      // rather than the O(slope*dz) error of integer-layer sampling)
      Real kgr = (z - dzmin_c)/dzm_c - 0.5;
      Real adk = fabs(kgr - static_cast<Real>(k0_c));
      Real arg = 2.0*M_PI*(n1_c*x/lx_c + n2_c*(y + qomt_c*x)/ly_c);
      Real ana = camp*pow(mu, adk)*cos(arg)
               + bgfac*(0.5*kgr*(kgr+1.0)
                        + 0.5*(static_cast<Real>(nz_c)-1.0-kgr)
                             *(static_cast<Real>(nz_c)-kgr));
      Real diff = phi(m,0,k,j,i) - ana;
      lmax = fmax(lmax, fabs(diff));
      lsq += SQR(diff);
    }, Kokkos::Max<Real>(dmax), Kokkos::Sum<Real>(dsq));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &dsq, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "# FFT-POISSON SHEET ERROR: max_rel= " << dmax/std::abs(camp)
                << " l2_rel= " << std::sqrt(dsq/ncells_tot)/std::abs(camp)
                << std::endl;
    }
  }

  // ---- analytic slab comparison (profile = slab, vert_bc = open) -----------------------
  if (slab && open_z) {
    // compare phi(z) along one column to four_pi_G*rho0*H^2*ln(cosh((z-zc)/H)),
    // both shifted to zero column-mean (potential defined up to a constant)
    int nz = nk;
    DvceArray1D<Real> col("fftp_col", nz);
    auto phi_d = phi;
    par_for("fftp_slabcol", DevExeSpace(), ks, ke,
    KOKKOS_LAMBDA(const int k) {
      col(k-ks) = phi_d(0,0,k,js,is);
    });
    auto col_h = Kokkos::create_mirror_view(col);
    Kokkos::deep_copy(col_h, col);
    // block 0 z-coordinates (host)
    Real b0_x3min = size.h_view(0).x3min, b0_x3max = size.h_view(0).x3max;
    Real ana_mean = 0.0, num_mean = 0.0;
    std::vector<Real> ana(nz);
    for (int k=0; k<nz; ++k) {
      Real z = CellCenterX(k, indcs.nx3, b0_x3min, b0_x3max);
      ana[k] = four_pi_G*rho0*SQR(slab_h)*std::log(std::cosh((z - zc)/slab_h));
      ana_mean += ana[k]/nz;
      num_mean += col_h(k)/nz;
    }
    Real err_max = 0.0, ana_amp = 0.0;
    for (int k=0; k<nz; ++k) {
      err_max = std::fmax(err_max, std::abs((col_h(k) - num_mean) - (ana[k] - ana_mean)));
      ana_amp = std::fmax(ana_amp, std::abs(ana[k] - ana_mean));
    }
    if (global_variable::my_rank == 0) {
      std::cout << "# FFT-POISSON SLAB ERROR: max_rel= " << err_max/ana_amp << std::endl;
    }
  }

  // ---- MG vs FFT cross-check (<problem> compare_fft = true) ---------------------------
  // With solver=multigrid, additionally run the independent FFT solver on the same
  // density (vert_bc selects its vertical BC; use "open" to cross-check mg_bc=slab
  // against the Koyama & Ostriker two-solve method) and report the mean-subtracted
  // difference.  The methods differ at the level of the KO09 continuum-kperp
  // screening weights and (when sheared) the roll remap, so the difference is small
  // but not machine zero.
#if FFT_ENABLED
  if (pin->GetOrAddBoolean("problem", "compare_fft", false) &&
      pmbp->pgrav->pmgd != nullptr) {
    DvceArray5D<Real> phi_mg("cmp_phimg", phi.extent(0), phi.extent(1),
                             phi.extent(2), phi.extent(3), phi.extent(4));
    Kokkos::deep_copy(phi_mg, phi);
    auto *pfft_cmp = new gravity::FFTGravitySolver(pmbp, pin);
    pfft_cmp->Solve(nullptr, 1);
    delete pfft_cmp;

    Real s1 = 0.0, s2 = 0.0;
    Kokkos::parallel_reduce("fftp_cmp_means",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &l1, Real &l2s) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      l1 += phi_mg(m,0,k,j,i);
      l2s += phi(m,0,k,j,i);
    }, Kokkos::Sum<Real>(s1), Kokkos::Sum<Real>(s2));
    Real m1 = s1/ncells_tot, m2 = s2/ncells_tot;

    Real dmax = 0.0, dsq = 0.0, refsq = 0.0;
    Kokkos::parallel_reduce("fftp_cmp_err",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &lmax, Real &lsq, Real &lref) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      Real diff = (phi_mg(m,0,k,j,i) - m1) - (phi(m,0,k,j,i) - m2);
      lmax = fmax(lmax, fabs(diff));
      lsq += SQR(diff);
      lref += SQR(phi(m,0,k,j,i) - m2);
    }, Kokkos::Max<Real>(dmax), Kokkos::Sum<Real>(dsq), Kokkos::Sum<Real>(refsq));

    Real ref_rms = std::sqrt(refsq/ncells_tot);
    if (global_variable::my_rank == 0) {
      std::cout << "# MG-VS-FFT DIFF: max= " << dmax/ref_rms
                << " l2= " << std::sqrt(dsq/ncells_tot)/ref_rms
                << " (relative to rms of the FFT solution)" << std::endl;
    }
    Kokkos::deep_copy(phi, phi_mg);
  }
#endif
}
