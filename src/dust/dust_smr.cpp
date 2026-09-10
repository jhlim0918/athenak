//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_smr.cpp
//! \brief Static-mesh-refinement support of the finest-level deposits (dust track,
//! particles + SMR).  A MeshBlock one level below the finest level scatters its particles
//! with the finest-level kernel into a 2x fine image of itself (2*ng ghost cells and 2*nx
//! active cells per direction, see DepositStencil); the image is volume-averaged onto the
//! block's own cells here, active zone and ghost shell alike, before the additive
//! exchange.  The exchange sends the fine-image ghost cells themselves to finer
//! neighbours (MeshBoundaryValuesDep), so a fine cell receives exactly the finest-kernel
//! weight of every nearby particle whatever block it lives in, and a coarse cell the
//! volume average of those weights: one consistent deposited field, conserved to
//! round-off.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "dust.hpp"

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ZeroImage
//! \brief Zero a fine image before a scatter (no-op without coarser blocks).

void DustGasDrag::ZeroImage(DvceArray5D<Real> &fimg) {
  if (!any_coarse) {return;}
  Kokkos::deep_copy(DevExeSpace(), fimg, 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::RestrictImage
//! \brief On blocks with rfac = r > 1, a(m,v,k,j,i) = mean of the r^d fine-image cells
//! of every block cell, ghosts included (the image and the block have the same ghost
//! depth in coarse units).  Blocks with rfac = 1 deposited into a directly, skipped.

void DustGasDrag::RestrictImage(DvceArray5D<Real> &fimg, DvceArray5D<Real> &a) {
  if (!any_coarse) {return;}
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nvar = a.extent_int(1);
  const int n1 = a.extent_int(4), n2 = a.extent_int(3), n3 = a.extent_int(2);
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto rf = rfac.d_view;
  par_for("dust_restrict_image", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, n2-1,
          0, n1-1,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    const int r = rf(m);
    if (r == 1) {return;}
    const int rj = multi_d ? r : 1, rk = three_d ? r : 1;
    const int fi = r*i, fj = multi_d ? r*j : j, fk = three_d ? r*k : k;
    Real sum = 0.0;
    for (int kk=0; kk<rk; ++kk) {
      for (int jj=0; jj<rj; ++jj) {
        for (int ii=0; ii<r; ++ii) {
          sum += fimg(m,v,fk+kk,fj+jj,fi+ii);
        }
      }
    }
    a(m,v,k,j,i) = sum/static_cast<Real>(r*rj*rk);
  });
}

} // namespace dust
