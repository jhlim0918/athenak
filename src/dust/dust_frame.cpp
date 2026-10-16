//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_frame.cpp
//! \brief The dust module in a 3D shearing box WITHOUT orbital advection.
//!
//! With <shearing_box> orbital_advection = true (FARGO, the default) the gas momenta are
//! perturbations about the background shear v_sh = -q Omega x; with it false they are the
//! full momenta.  The dust module works throughout in the shear-relative frame: particle
//! velocities (dust_push.cpp), the deposited dust momenta, the gas velocity u* gathered by
//! the particles, the drag solver's unknowns and the copy exchanges and shear remaps of
//! those fields.  Instead of converting at every place a gas velocity is read, every task
//! that reads or writes the gas momenta holds a GasFrameGuard, which shifts the gas into
//! the shear-relative frame on entry and back on exit:
//!     M2' = M2 + rho t,   E' = E + M2 t + rho t^2/2,   t = -v_sh (to relative), +v_sh (back)
//! (the energy shift keeps the internal energy fixed; isothermal runs have no IEN).  The
//! density is untouched, so the two shifts are exact inverses up to round-off, and every
//! dust operation inside the guard -- additive momentum changes, frictional heating in the
//! relative frame, and the implicit solves that assign new velocities -- acts on the gas
//! exactly as it would with orbital advection.  The shift covers ghost cells at their own
//! x, so the shear-periodic momentum offset of the full-frame ghosts (qOmega Lx rho)
//! becomes the continuous relative-frame field the dust exchanges and remaps expect.
//! Inactive (bitwise no-op) with orbital advection or outside a 3D shearing box.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "dust/dust.hpp"

namespace dust {

void DustGasDrag::ShiftGasFrame(bool to_relative) {
  hydro::Hydro *phyd = pmy_pack->phydro;
  if (phyd == nullptr) {return;}
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int is = indcs.is, nx1 = indcs.nx1;
  const Real qo = qshear*omega0;
  const bool ideal = phyd->peos->eos_data.is_ideal;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const Real sgn = to_relative ? 1.0 : -1.0;     // t = sgn * q Omega x  (= -+ v_sh)
  auto shift = [&](DvceArray5D<Real> &u) {
    const int n3 = u.extent_int(2), n2 = u.extent_int(3), n1 = u.extent_int(4);
    par_for("dust_gas_frame", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      const Real x = x1min + (static_cast<Real>(i - is) + 0.5)*(x1max - x1min)/nx1;
      const Real t = sgn*qo*x;
      const Real rho = u(m,IDN,k,j,i);
      if (ideal) {u(m,IEN,k,j,i) += u(m,IM2,k,j,i)*t + 0.5*rho*t*t;}
      u(m,IM2,k,j,i) += rho*t;
    });
  };
  shift(phyd->u0);
  if (phyd->u1.size() == phyd->u0.size()) {shift(phyd->u1);}
}

} // namespace dust
