//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file streaming_strat.cpp
//! \brief Problem generator for the nonlinear streaming instability in a vertically
//! stratified 3D shearing box (Johansen, Youdin & Mac Low 2009; Bai & Stone 2010b; Yang
//! & Johansen 2014), the AthenaK counterpart of Athena-C's par_strat3d.c.
//!
//! Isothermal gas in vertical hydrostatic equilibrium, rho_g = rho0 exp(-z^2/2H^2) with
//! H = c_s/Omega, feels the linearized stellar gravity -Omega^2 z (<shearing_box>/
//! stratified = true, which also applies it to the dust particles) and the radial
//! pressure-gradient mimic a_x = 2 Omega etavk on the gas alone (<hydro_srcterms>/
//! const_accel, as in dust_nsh), where etavk = Pi c_s is the azimuthal velocity
//! reduction of dust-free gas.  Periodic x3 boundaries for gas and particles are
//! consistent with the symmetric equilibrium (Yang & Johansen 2014, sec. 2.1).
//!
//! The dust is <particles>/ppc particles per cell in total, split evenly among the
//! <dust>/nspecies species (interleaved round-robin within every MeshBlock), uniform in
//! (x,y) and in z either uniform over the box (<problem>/dust_hz <= 0, Yang & Johansen
//! 2014) or a Gaussian of width dust_hz truncated to the box (Bai & Stone 2010b).  The
//! solid-to-gas column ratio Z = Sigma_p/Sigma_g follows Yang & Johansen (2014, sec.
//! 2.2): Sigma_g = sqrt(2 pi) H rho0 is the column of the full (untruncated) disk, so
//! species s has total mass Z_s sqrt(2 pi) H rho0 Lx Ly whatever Lz is (<problem>/dust_Z,
//! split evenly unless dust_Z_s are given).  Per-block counts follow the dust mass in
//! each block's z range and positions are hashed from the global block id and the index
//! in the block (<problem>/random_seed), so the initial state does not depend on the
//! MPI decomposition.
//!
//! Initial velocities: particles at rest in the shearing frame; gas at azimuthal
//! velocity <problem>/gas_vy0 (default 0, Yang & Johansen 2014; -etavk is the dust-free
//! equilibrium, i.e. Athena-C par_strat3d ipert = 1).

// C++ headers
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// AthenaK headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "shearing_box/shearing_box.hpp"
#include "srcterms/srcterms.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "pgen/pgen.hpp"

namespace {

// deterministic integer hash -> uniform Real in [0, 1): splitmix64 finalizer
KOKKOS_INLINE_FUNCTION
Real HashU01(int64_t gid, int64_t q, int64_t c, int64_t trial, int64_t seed) {
  uint64_t x = static_cast<uint64_t>(seed);
  x += 0x9e3779b97f4a7c15ULL*static_cast<uint64_t>(gid + 1);
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x += 0x9e3779b97f4a7c15ULL*static_cast<uint64_t>(q + 1);
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  x += 0x9e3779b97f4a7c15ULL*static_cast<uint64_t>(c + 1);
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x += 0x9e3779b97f4a7c15ULL*static_cast<uint64_t>(trial + 1);
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return static_cast<Real>(x >> 11)/9007199254740992.0;  // 53-bit mantissa
}

void Fatal(const char *file, int line, const std::string &msg) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::StreamingStrat()
//! \brief Stratified streaming-instability box: gas in hydrostatic equilibrium, dust
//! particles at rest (see the file header)

void ProblemGenerator::StreamingStrat(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmy_mesh_->three_d) {
    Fatal(__FILE__, __LINE__, "streaming_strat works in 3D only");
  }
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr) {
    Fatal(__FILE__, __LINE__, "streaming_strat needs <hydro>, <particles> and <dust>");
  }
  if (pmbp->phydro->psbox_u == nullptr || !pmbp->phydro->psbox_u->is_stratified) {
    Fatal(__FILE__, __LINE__, "streaming_strat needs <shearing_box> stratified = true");
  }
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  if (eos.is_ideal) {
    Fatal(__FILE__, __LINE__, "streaming_strat needs an isothermal EOS");
  }
  Real omega0 = pin->GetReal("shearing_box", "omega0");
  Real cs = eos.iso_cs;
  Real hgas = cs/omega0;
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real etavk = pin->GetReal("problem", "etavk");
  Real gas_vy0 = pin->GetOrAddReal("problem", "gas_vy0", 0.0);

  // the radial pressure-gradient mimic must act on the gas alone through const_accel
  {
    auto *psrc = pmbp->phydro->psrc;
    Real ax = 2.0*omega0*etavk;
    bool ok = (psrc != nullptr && psrc->const_accel && psrc->const_accel_dir == 1 &&
               std::fabs(psrc->const_accel_val - ax) <= 1.0e-12*std::fabs(ax));
    if (!ok) {
      Fatal(__FILE__, __LINE__, "streaming_strat needs <hydro_srcterms> const_accel = "
            "true, const_accel_dir = 1, const_accel_val = 2*omega0*etavk = " +
            std::to_string(ax));
    }
  }

  // gas: isothermal hydrostatic equilibrium (all cells, ghosts included)
  auto &indcs = pmy_mesh_->mb_indcs;
  int ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng, n2 = indcs.nx2 + 2*ng, n3 = indcs.nx3 + 2*ng;
  int nx3 = indcs.nx3, ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  par_for("strat_gas", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real den = rho0*exp(-0.5*SQR(z/hgas));
    u0(m,IDN,k,j,i) = den;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = den*gas_vy0;
    u0(m,IM3,k,j,i) = 0.0;
  });

  // dust: per-block counts from the dust mass in each block, rounded to nspecies
  particles::Particles *ppar = pmbp->ppart;
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min;
  Real ly = msize.x2max - msize.x2min;
  Real lz = msize.x3max - msize.x3min;
  Real ppc = pin->GetOrAddReal("particles", "ppc", 1.0);
  int nsp = pmbp->pdust->nspecies;
  Real dust_Z = pin->GetOrAddReal("problem", "dust_Z", 0.02);
  std::vector<Real> zs(nsp);
  for (int s=0; s<nsp; ++s) {
    zs[s] = pin->GetOrAddReal("problem", "dust_Z_" + std::to_string(s+1), dust_Z/nsp);
  }
  Real dust_hz = pin->GetOrAddReal("problem", "dust_hz", 0.0);
  int64_t seed = pin->GetOrAddInteger("problem", "random_seed", 1);
  Real n_target = ppc*static_cast<Real>(pmy_mesh_->mesh_indcs.nx1)
                     *static_cast<Real>(pmy_mesh_->mesh_indcs.nx2)
                     *static_cast<Real>(pmy_mesh_->mesh_indcs.nx3);
  bool gauss = (dust_hz > 0.0);
  Real s2 = gauss ? dust_hz*std::sqrt(2.0) : 1.0;
  Real znorm = gauss ? 0.5*(std::erf(msize.x3max/s2) - std::erf(msize.x3min/s2)) : lz;
  std::vector<int> off(nmb + 1, 0);
  for (int m=0; m<nmb; ++m) {
    Real area = (size.h_view(m).x1max - size.h_view(m).x1min)
               *(size.h_view(m).x2max - size.h_view(m).x2min);
    Real zl = size.h_view(m).x3min, zu = size.h_view(m).x3max;
    Real zfrac = gauss ? 0.5*(std::erf(zu/s2) - std::erf(zl/s2))/znorm : (zu - zl)/lz;
    int n_m = nsp*static_cast<int>(std::floor(n_target*(area/(lx*ly))*zfrac/nsp + 0.5));
    off[m+1] = off[m] + n_m;
  }
  int npart = off[nmb];
  int64_t ntot = npart;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &ntot, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
  // the Mesh particle counts and the particle tags are 32-bit ints
  if (ntot > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    Fatal(__FILE__, __LINE__, "streaming_strat: " + std::to_string(ntot) + " particles "
          "exceed the 32-bit particle count and tags of AthenaK (2^31 - 1); lower "
          "<particles>/ppc");
  }
  // species s: ntot/nsp particles sharing the mass Z_s Sigma_g Lx Ly
  Real sigma_g = std::sqrt(2.0*M_PI)*hgas*rho0;
  DualArray1D<Real> mps("strat_mp", nsp);
  for (int s=0; s<nsp; ++s) {
    mps.h_view(s) = zs[s]*sigma_g*lx*ly*static_cast<Real>(nsp)
                    /static_cast<Real>(std::max<int64_t>(ntot, 1));
  }
  mps.modify_host();
  mps.sync_device();

  // replace the placeholder particles allocated from <particles>/ppc
  ppar->nprtcl_thispack = npart;
  Kokkos::realloc(ppar->prtcl_rdata, ppar->nrdata, std::max(npart, 1));
  Kokkos::realloc(ppar->prtcl_idata, ppar->nidata, std::max(npart, 1));
  DvceArray1D<int> d_off("strat_off", nmb + 1);
  {
    auto h_off = Kokkos::create_mirror_view(d_off);
    for (int m=0; m<=nmb; ++m) h_off(m) = off[m];
    Kokkos::deep_copy(d_off, h_off);
  }
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto gids = pmbp->gids;
  auto &taus_ = pmbp->pdust->taus;
  Real hz = dust_hz;
  par_for("strat_dust", DevExeSpace(), 0, (npart-1), KOKKOS_LAMBDA(const int p) {
    // owning block by binary search over the offsets
    int lo = 0, hi = nmb - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1)/2;
      if (d_off(mid) <= p) {lo = mid;} else {hi = mid - 1;}
    }
    int m = lo;
    int64_t q = p - d_off(m);
    int64_t gid = gids + m;
    Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;
    pr(IPX,p) = x1min + HashU01(gid, q, 0, 0, seed)*(x1max - x1min);
    pr(IPY,p) = x2min + HashU01(gid, q, 1, 0, seed)*(x2max - x2min);
    Real z = x3min + HashU01(gid, q, 2, 0, seed)*(x3max - x3min);
    if (hz > 0.0) {
      // Gaussian truncated to the block: Box-Muller from hashed uniforms, rejection
      for (int trial=0; trial<256; ++trial) {
        Real u1 = fmax(HashU01(gid, q, 3, trial, seed), 1.0e-300);
        Real u2 = HashU01(gid, q, 4, trial, seed);
        Real zt = hz*sqrt(-2.0*log(u1))*cos(6.283185307179586*u2);
        if (zt >= x3min && zt < x3max) {z = zt; break;}
      }
    }
    pr(IPZ,p) = z;
    int sp = static_cast<int>(q % nsp);
    pi(PGID,p) = static_cast<int>(gid);
    pi(PSP,p) = sp;
    pr(IPTS,p) = taus_.d_view(sp);
    pr(IPM,p) = mps.d_view(sp);
    pr(IPVX,p) = 0.0; pr(IPVY,p) = 0.0; pr(IPVZ,p) = 0.0;
    pr(IPRX,p) = 0.0; pr(IPRY,p) = 0.0; pr(IPRZ,p) = 0.0;
  });

  // Mesh bookkeeping and tags
  Mesh *pm = pmy_mesh_;
  pm->nprtcl_thisrank = npart;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&(pm->nprtcl_thisrank), 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT,
                MPI_COMM_WORLD);
#else
  pm->nprtcl_eachrank[0] = npart;
#endif
  pm->nprtcl_total = 0;
  for (int r=0; r<global_variable::nranks; ++r) {
    pm->nprtcl_total += pm->nprtcl_eachrank[r];
  }
  ppar->CreateParticleTags(pin);

  if (global_variable::my_rank == 0) {
    Real mgas_box = rho0*lx*ly*std::sqrt(2.0*M_PI)*hgas
                   *0.5*(std::erf(msize.x3max/(std::sqrt(2.0)*hgas))
                         - std::erf(msize.x3min/(std::sqrt(2.0)*hgas)));
    std::cout << "streaming_strat: H = " << hgas << ", Sigma_g = sqrt(2 pi) H rho0 = "
              << sigma_g << ", gas mass in box = " << mgas_box << std::endl
              << "streaming_strat: " << pm->nprtcl_total << " particles (target "
              << static_cast<int64_t>(n_target) << "), "
              << (gauss ? "Gaussian h_z = " + std::to_string(dust_hz) :
                          std::string("uniform in z")) << std::endl;
    Real ztot = 0.0;
    for (int s=0; s<nsp; ++s) {
      ztot += zs[s];
      std::cout << "streaming_strat:   species " << s+1 << ": stopping time "
                << taus_.h_view(s) << ", Z = " << zs[s] << ", "
                << pm->nprtcl_total/nsp << " particles of mass " << mps.h_view(s)
                << std::endl;
    }
    std::cout << "streaming_strat: dust/gas mass in box = "
              << ztot*sigma_g*lx*ly/mgas_box << ", etavk = " << etavk
              << " (Pi = " << etavk/cs << "), gas_vy0 = " << gas_vy0 << std::endl;
  }
}
