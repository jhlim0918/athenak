//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_interface_wave.cpp
//! \brief Static Poisson check of the self-gravity force at mesh-refinement level jumps:
//! the "poisson" mode of R. Li's self_gravity_wave test (athenak-dust amrSG_dev,
//! src/pgen/tests/self_gravity_wave.cpp) for our multigrid solver.
//!
//! Density rho = rho0 [1 + amp S cos(k.x + phase)], S = prod_d sinc(k_d dx_d/2) (the cell
//! average of the cosine), k = 2 pi (nwx/Lx, nwy/Ly, nwz/Lz). The exact potential is
//! phi = -(4 pi G rho0 amp/k^2) cos(k.x + phase) up to a constant, and g = -grad phi. One
//! solve, then the errors of phi (both sides mean-subtracted) and of the force the gas
//! source term applies (SourceTerms::AddSelfGravity: g_d = -(phi_{+1} - phi_{-1})/(2 dx_d),
//! through the ghost zones the solver filled) against the exact g at the cell centers.
//! "Interface" cells: the first/last active cell of a block on a face whose neighbor is
//! on another level (as in self_gravity_wave). Our multigrid needs logically cubic 3-D
//! MeshBlocks, so the 2-D test of self_gravity_wave (nwz = 0) is run as a 3-D slab one
//! block thick in x3 with periodic x3 (the solution does not depend on x3).

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "hydro/hydro.hpp"
#include "gravity/gravity.hpp"
#include "pgen/pgen.hpp"

namespace {
KOKKOS_INLINE_FUNCTION
Real SincIW(Real v) { return fabs(v) < 1.0e-10 ? 1.0 : sin(v)/v; }
}  // namespace

void ProblemGenerator::MGInterfaceWave(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pgrav == nullptr || pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in MGInterfaceWave: needs <hydro> and <gravity>" << std::endl;
    exit(EXIT_FAILURE);
  }
  const Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real amp = pin->GetOrAddReal("problem", "amp", 0.1);
  const Real phase = pin->GetOrAddReal("problem", "phase", 0.37);
  const int nwx = pin->GetOrAddInteger("problem", "nwx", 2);
  const int nwy = pin->GetOrAddInteger("problem", "nwy", 1);
  const int nwz = pin->GetOrAddInteger("problem", "nwz", 0);
  const auto &ms = pmy_mesh_->mesh_size;
  const Real x0 = ms.x1min, y0 = ms.x2min, z0 = ms.x3min;
  const Real kx = 2.0*M_PI*nwx/(ms.x1max - ms.x1min);
  const Real ky = 2.0*M_PI*nwy/(ms.x2max - ms.x2min);
  const Real kz = 2.0*M_PI*nwz/(ms.x3max - ms.x3min);
  const Real k2 = kx*kx + ky*ky + kz*kz;
  const Real coef = four_pi_G*rho0*amp/k2;

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;
  par_for("mgiw_init", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const auto s = size.d_view(m);
    Real x = s.x1min + (i - is + 0.5)*s.dx1 - x0;
    Real y = s.x2min + (j - js + 0.5)*s.dx2 - y0;
    Real z = s.x3min + (k - ks + 0.5)*s.dx3 - z0;
    Real avg = SincIW(0.5*kx*s.dx1)*SincIW(0.5*ky*s.dx2)*SincIW(0.5*kz*s.dx3);
    Real d = rho0*(1.0 + amp*avg*cos(kx*x + ky*y + kz*z + phase));
    u0(m,IDN,k,j,i) = d;
    u0(m,IM1,k,j,i) = 0.0; u0(m,IM2,k,j,i) = 0.0; u0(m,IM3,k,j,i) = 0.0;
    w0(m,IDN,k,j,i) = d;
    w0(m,IVX,k,j,i) = 0.0; w0(m,IVY,k,j,i) = 0.0; w0(m,IVZ,k,j,i) = 0.0;
  });

  pmbp->pgrav->Solve(nullptr, 1);

  auto phi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pgrav->phi);
  // sums: 0 vol, 1 vol*exact, 2 vol*num (pass 0); 3 phi err^2, 4 exact^2, 5 force err^2,
  // 6 force^2, 7 interface vol, 8 interface force err^2
  Real sums[9] = {};
  Real maxima[4] = {};   // phi err, force err, interface force err, max |g| exact
  for (int pass = 0; pass < 2; ++pass) {
    for (int m = 0; m < nmb; ++m) {
      const auto s = size.h_view(m);
      const Real vol = s.dx1*s.dx2*s.dx3;
      bool iface[6] = {};
      for (int d = 0; d < 3; ++d) {
        for (int side = 0; side < 2; ++side) {
          int off[3] = {0, 0, 0}; off[d] = 2*side - 1;
          for (int f1 = 0; f1 < 2; ++f1) {
            for (int f2 = 0; f2 < 2; ++f2) {
              int slot = NeighborIndex(off[0], off[1], off[2], f1, f2);
              auto nb = pmbp->pmb->nghbr.h_view(m, slot);
              iface[2*d + side] |= (nb.gid >= 0 && nb.lev != pmbp->pmb->mb_lev.h_view(m));
            }
          }
        }
      }
      for (int k = ks; k <= ke; ++k) for (int j = js; j <= je; ++j) for (int i = is; i <= ie; ++i) {
        Real x = s.x1min + (i - is + 0.5)*s.dx1 - x0;
        Real y = s.x2min + (j - js + 0.5)*s.dx2 - y0;
        Real z = s.x3min + (k - ks + 0.5)*s.dx3 - z0;
        Real th = kx*x + ky*y + kz*z + phase;
        Real exact = -coef*cos(th);
        if (pass == 0) {sums[0] += vol; sums[1] += vol*exact; sums[2] += vol*phi(m,0,k,j,i); continue;}
        Real err = (phi(m,0,k,j,i) - sums[2]/sums[0]) - (exact - sums[1]/sums[0]);
        Real g[3] = {-(phi(m,0,k,j,i+1) - phi(m,0,k,j,i-1))/(2.0*s.dx1),
                     -(phi(m,0,k,j+1,i) - phi(m,0,k,j-1,i))/(2.0*s.dx2),
                     -(phi(m,0,k+1,j,i) - phi(m,0,k-1,j,i))/(2.0*s.dx3)};
        Real kv[3] = {kx, ky, kz};
        Real fe = 0.0, fx = 0.0;
        for (int d = 0; d < 3; ++d) {
          Real ge = -coef*kv[d]*sin(th);
          fe += SQR(g[d] - ge); fx += ge*ge;
        }
        bool inter = (i == is && iface[0]) || (i == ie && iface[1]) || (j == js && iface[2]) ||
                     (j == je && iface[3]) || (k == ks && iface[4]) || (k == ke && iface[5]);
        sums[3] += vol*err*err; sums[4] += vol*SQR(exact - sums[1]/sums[0]);
        sums[5] += vol*fe; sums[6] += vol*fx;
        if (inter) {sums[7] += vol; sums[8] += vol*fe;}
        maxima[0] = std::max(maxima[0], std::abs(err));
        maxima[1] = std::max(maxima[1], std::sqrt(fe));
        if (inter) maxima[2] = std::max(maxima[2], std::sqrt(fe));
        maxima[3] = std::max(maxima[3], std::sqrt(fx));
      }
    }
#if MPI_PARALLEL_ENABLED
    if (pass == 0) MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums+3, 6, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, maxima, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(10)
      << "# MG_INTERFACE_WAVE {\"blocks\":" << pmy_mesh_->nmb_total
      << ",\"phi_relative_l2\":" << std::sqrt(sums[3]/sums[4])
      << ",\"phi_linf\":" << maxima[0]
      << ",\"acceleration_relative_l2\":" << std::sqrt(sums[5]/sums[6])
      << ",\"acceleration_linf\":" << maxima[1]
      << ",\"interface_volume\":" << sums[7]
      << ",\"interface_acceleration_l2\":" << (sums[7] > 0 ? std::sqrt(sums[8]/sums[7]) : 0.0)
      << ",\"interface_acceleration_linf\":" << maxima[2]
      << ",\"peak_force\":" << maxima[3] << "}" << std::endl;
  }
}
