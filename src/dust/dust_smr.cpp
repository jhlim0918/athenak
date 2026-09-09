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
//! \brief On blocks with rfac = 2, a(m,v,k,j,i) = mean of the 2^d fine-image cells of
//! every block cell, ghosts included (the image and the block have the same ghost depth
//! in coarse units).  Blocks with rfac = 1 deposited into a directly and are skipped.

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
    if (rf(m) == 1) {return;}
    const int fi = 2*i;
    if (three_d) {
      const int fj = 2*j, fk = 2*k;
      a(m,v,k,j,i) = 0.125*(fimg(m,v,fk  ,fj  ,fi) + fimg(m,v,fk  ,fj  ,fi+1)
                          + fimg(m,v,fk  ,fj+1,fi) + fimg(m,v,fk  ,fj+1,fi+1)
                          + fimg(m,v,fk+1,fj  ,fi) + fimg(m,v,fk+1,fj  ,fi+1)
                          + fimg(m,v,fk+1,fj+1,fi) + fimg(m,v,fk+1,fj+1,fi+1));
    } else if (multi_d) {
      const int fj = 2*j;
      a(m,v,k,j,i) = 0.25*(fimg(m,v,k,fj  ,fi) + fimg(m,v,k,fj  ,fi+1)
                         + fimg(m,v,k,fj+1,fi) + fimg(m,v,k,fj+1,fi+1));
    } else {
      a(m,v,k,j,i) = 0.5*(fimg(m,v,k,j,fi) + fimg(m,v,k,j,fi+1));
    }
  });
}

} // namespace dust
