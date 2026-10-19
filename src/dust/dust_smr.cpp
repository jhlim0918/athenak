//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_smr.cpp
//! \brief Mesh-refinement support of the finest-level deposits (dust track, particles +
//! SMR/AMR).  Every MeshBlock deposits with the kernel of the finest level present,
//! binned onto its own cells (DepositStencil), so a block r times coarser carries no
//! fine image of itself.  The additive exchange sends a coarse block's ghost deposits to
//! a FINER neighbour at that neighbour's resolution (MeshBoundaryValuesDep), i.e. as the
//! 2^d half cells of each ghost cell; a block with a finer neighbour therefore also bins
//! the deposits of its edge particles onto a half-cell halo image (2 x the block's
//! resolution, whatever r is).  A fine cell thus receives exactly the finest-kernel
//! weight of every nearby particle whatever block it lives in, and a coarse cell the
//! volume average of those weights: one consistent deposited field, conserved to
//! round-off.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "dust.hpp"

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ZeroImage
//! \brief Zero the halo image of every block with a finer neighbour (halo(m) = 1) before
//! a scatter; the other blocks never write or send theirs.  No-op without such blocks.

void DustGasDrag::ZeroImage(DvceArray5D<Real> &fimg) {
  if (!any_halo) {return;}
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nvar = fimg.extent_int(1);
  const int n1 = fimg.extent_int(4), n2 = fimg.extent_int(3), n3 = fimg.extent_int(2);
  auto hl = halo.d_view;
  par_for("dust_zero_halo", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    if (hl(m) == 0) {return;}
    fimg(m,v,k,j,i) = 0.0;
  });
}

} // namespace dust
