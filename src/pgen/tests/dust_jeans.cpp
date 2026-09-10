//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_jeans.cpp
//! \brief The dusty Jeans instability of Krapp et al. (2024, ApJS 271, 7, sec. 3.5) with
//! superparticle dust (dust track, Phase 4c validation of multigrid + particles).
//! A uniform isothermal gas (rho0, cs) and a uniform dust lattice (species s with
//! dust-to-gas ratio eps_s and stopping time taus_s) in a periodic box, perturbed along
//! x with the eigenmode of the linearized gas + dust + Poisson system (Jeans swindle:
//! the periodic solve subtracts the mean).  The perturbation amplitudes and phases of
//! every component are <problem> parameters, produced by scripts/analysis/dust_jeans.py
//! from the eigenproblem, so the run starts on the fastest-growing mode; the dust
//! density perturbation is carried by the particle MASSES (exact, noise-free) and the
//! dust velocity by the lattice velocities.
//! History (user_hist): the cos/sin projections (2/V) sum q cos(kx) dV of the gas
//! density and, per species, of the particle mass: the mode amplitude is
//! sqrt(c^2 + s^2) and its growth rate is the test.  Requires <gravity> (periodic
//! multigrid or fft), <hydro_srcterms> self_gravity = true, <dust> gravity = true,
//! <particles> ppc = nspecies.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "gravity/gravity.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "pgen/pgen.hpp"

namespace {
Real jeans_kx = 0.0;   // mode wavenumber, shared with the history function
void DustJeansHistory(HistoryData *pdata, Mesh *pm);
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::DustJeans()

void ProblemGenerator::DustJeans(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->pgrav == nullptr || pmbp->ppart == nullptr ||
      pmbp->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust_jeans requires <hydro>, <gravity>, <particles> and <dust>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(pmy_mesh_->three_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust_jeans requires a 3D mesh (the gravity solvers do)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  pmbp->pgrav->four_pi_G = four_pi_G;
  if (pmbp->pgrav->pmgd != nullptr) {pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);}
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min;
  int kmode = pin->GetOrAddInteger("problem", "kmode", 1);
  jeans_kx = 2.0*M_PI*static_cast<Real>(kmode)/lx;
  user_hist_func = DustJeansHistory;
  if (restart) return;

  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real rho_amp = pin->GetOrAddReal("problem", "rho_amp", 1.0e-4);
  Real vg_amp = pin->GetOrAddReal("problem", "vg_amp", 0.0);
  Real vg_phase = pin->GetOrAddReal("problem", "vg_phase", 0.0);
  int nspec = pmbp->pdust->nspecies;
  // per species: eps, rhod_amp, rhod_phase, vd_amp, vd_phase
  DualArray2D<Real> spdat("jeans_spdat", nspec, 5);
  for (int s = 0; s < nspec; ++s) {
    std::string n = std::to_string(s+1);
    spdat.h_view(s,0) = pin->GetOrAddReal("problem", "eps_"+n, 0.01);
    spdat.h_view(s,1) = pin->GetOrAddReal("problem", "rhod_amp_"+n, 0.0);
    spdat.h_view(s,2) = pin->GetOrAddReal("problem", "rhod_phase_"+n, 0.0);
    spdat.h_view(s,3) = pin->GetOrAddReal("problem", "vd_amp_"+n, 0.0);
    spdat.h_view(s,4) = pin->GetOrAddReal("problem", "vd_phase_"+n, 0.0);
  }
  spdat.template modify<HostMemSpace>();
  spdat.template sync<DevExeSpace>();
  Real kx = jeans_kx;

  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  bool is_ideal = eos.is_ideal;
  Real gm1 = eos.gamma - 1.0;
  Real cs = eos.iso_cs;
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;

  par_for("jeans_gas", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real rho = rho0 + rho_amp*cos(kx*x);
    Real vx = vg_amp*cos(kx*x + vg_phase);
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho*vx;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    if (is_ideal) {u0(m,IEN,k,j,i) = rho*cs*cs/gm1 + 0.5*rho*vx*vx;}
  });

  // dust lattice: nspecies particles per cell at the cell centre; the density
  // perturbation is the mass modulation, the velocity perturbation the lattice velocity
  particles::Particles *ppar = pmbp->ppart;
  int npart = ppar->nprtcl_thispack;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto gids = pmbp->gids;
  auto &taus_ = pmbp->pdust->taus;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  int npart_permb = npart/nmb;
  int ppc_int = npart_permb/ncells;
  if (npart_permb != ppc_int*ncells || ppc_int != nspec) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust_jeans requires <particles>/ppc = <dust>/nspecies (one particle of "
              << "each species per cell)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int lnx1 = indcs.nx1, lnx2 = indcs.nx2;
  par_for("jeans_part", DevExeSpace(), 0, (npart-1), KOKKOS_LAMBDA(const int p) {
    int m = p/npart_permb;
    if (m > nmb-1) {m = nmb-1;}
    int q = p - m*npart_permb;
    int c = q/ppc_int;
    int s = q % ppc_int;
    int i = c % lnx1;
    int j = (c/lnx1) % lnx2;
    int k = c/(lnx1*lnx2);
    Real x = CellCenterX(i, lnx1, size.d_view(m).x1min, size.d_view(m).x1max);
    pr(IPX,p) = x;
    pr(IPY,p) = CellCenterX(j, lnx2, size.d_view(m).x2min, size.d_view(m).x2max);
    pr(IPZ,p) = CellCenterX(k, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    pi(PGID,p) = gids + m;
    pi(PSP,p) = s;
    pr(IPTS,p) = taus_.d_view(s);
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real eps = spdat.d_view(s,0);
    Real rhod = eps*rho0 + spdat.d_view(s,1)*cos(kx*x + spdat.d_view(s,2));
    pr(IPM,p) = rhod*vol;
    pr(IPVX,p) = spdat.d_view(s,3)*cos(kx*x + spdat.d_view(s,4));
    pr(IPVY,p) = 0.0; pr(IPVZ,p) = 0.0;
    pr(IPRX,p) = 0.0; pr(IPRY,p) = 0.0; pr(IPRZ,p) = 0.0;
  });
  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void DustJeansHistory()
//! \brief cos/sin projections of the gas density and of each species' mass on the mode

void DustJeansHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  int nspec = pmbp->pdust->nspecies;
  int nsout = std::min(nspec, (NHISTORY_VARIABLES - 2)/2);
  pdata->nhist = 2 + 2*nsout;
  pdata->label[0] = "rhog_c";
  pdata->label[1] = "rhog_s";
  for (int s = 0; s < nsout; ++s) {
    pdata->label[2+2*s] = "rhod" + std::to_string(s+1) + "_c";
    pdata->label[3+2*s] = "rhod" + std::to_string(s+1) + "_s";
  }
  auto &msize = pm->mesh_size;
  Real vbox = (msize.x1max - msize.x1min)*(msize.x2max - msize.x2min)
             *(msize.x3max - msize.x3min);
  Real norm = 2.0/vbox;
  Real kx = jeans_kx;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  Real gc = 0.0, gs = 0.0;
  Kokkos::parallel_reduce("jeans_hist_gas",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &c_, Real &s_) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    c_ += u0(m,IDN,k,j,i)*cos(kx*x)*vol;
    s_ += u0(m,IDN,k,j,i)*sin(kx*x)*vol;
  }, Kokkos::Sum<Real>(gc), Kokkos::Sum<Real>(gs));
  pdata->hdata[0] = norm*gc;
  pdata->hdata[1] = norm*gs;

  particles::Particles *ppar = pmbp->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  for (int s = 0; s < nsout; ++s) {
    Real dc = 0.0, ds = 0.0;
    Kokkos::parallel_reduce("jeans_hist_dust",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int &p, Real &c_, Real &s_) {
      if (pi(PSP,p) == s) {
        c_ += pr(IPM,p)*cos(kx*pr(IPX,p));
        s_ += pr(IPM,p)*sin(kx*pr(IPX,p));
      }
    }, Kokkos::Sum<Real>(dc), Kokkos::Sum<Real>(ds));
    pdata->hdata[2+2*s] = norm*dc;
    pdata->hdata[3+2*s] = norm*ds;
  }
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) {pdata->hdata[n] = 0.0;}
  return;
}

}  // namespace
