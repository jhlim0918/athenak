//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gravity.cpp
//! \brief implementation of functions in class Gravity

// C headers

// C++ headers
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "gravity.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "mg_gravity.hpp"
#include "../multigrid/multigrid.hpp"

#include "config.hpp"
#if FFT_ENABLED
#include "fft_gravity.hpp"
#endif

namespace gravity { // NOLINT (build/namespace)
//! constructor, initializes data structures and parameters
//-------------------------------------------------------------------------------------
//! \fn Gravity::Gravity(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Gravity constructor
Gravity::Gravity(MeshBlockPack *pmbp, ParameterInput *pin):
    pmy_pack(pmbp),
    phi("phi",1,1,1,1,1),
    coarse_phi("coarse",1,1,1,1,1),
    def("defect",1,1,1,1,1),
    rho_extra("rho_extra",1,1,1,1,1),
    rho_total("rho_total",1,1,1,1,1),
    four_pi_G(-1.0),
    output_defect(false),
    fill_ghost(false) {
    four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G",-1.0);
    output_defect = pin->GetOrAddBoolean("gravity", "output_defect", false);
    fill_ghost = pin->GetOrAddBoolean("gravity", "fill_ghost", true);

    if (four_pi_G == 0.0) {
        std::cout << "### FATAL ERROR in Gravity::Gravity" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
        exit(EXIT_FAILURE);
    }

    // create the selected Poisson solver: multigrid (default) or fft
    pmgd = nullptr;
    pmg = nullptr;
    pfft = nullptr;
    std::string solver = pin->GetOrAddString("gravity", "solver", "multigrid");
    if (solver == "multigrid") {
        // The driver allocates multigrid instances for root level and meshblock levels
        pmgd = new MGGravityDriver(pmbp, pin);
    } else if (solver == "fft") {
#if FFT_ENABLED
        pfft = new FFTGravitySolver(pmbp, pin);
#else
        std::cout << "### FATAL ERROR in Gravity::Gravity" << std::endl
        << "<gravity> solver = fft requires building with -D Athena_ENABLE_FFT=ON"
        << std::endl;
        exit(EXIT_FAILURE);
#endif
    } else {
        std::cout << "### FATAL ERROR in Gravity::Gravity" << std::endl
        << "<gravity> solver = '" << solver << "' not recognized "
        << "(must be 'multigrid' or 'fft')" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Enroll CellCenteredBoundaryVariable object
    //gbvar.bvar_index = pmb->pbval->bvars.size();
    //pmb->pbval->bvars.push_back(&gbvar);
    //pmb->pbval->pgbvar = &gbvar;
    int nmb = pmy_pack->nmb_thispack;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(phi, nmb, 1, ncells3, ncells2, ncells1);
}

//----------------------------------------------------------------------------------------
//! \fn Gravity::~Gravity()
//! \brief Gravity destructor
Gravity::~Gravity() {
    delete pmg;
#if FFT_ENABLED
    delete pfft;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn Gravity::RegisterExtraDensity()
//! \brief alias an additional density array into the Poisson source (see gravity.hpp)
void Gravity::RegisterExtraDensity(const DvceArray5D<Real> &rho) {
    rho_extra = rho;
    has_extra_density = true;
    int nmb = pmy_pack->nmb_thispack;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(rho_total, nmb, 1, ncells3, ncells2, ncells1);
}

const DvceArray5D<Real>& Gravity::SourceArray() const {
    if (has_extra_density) return rho_total;
    return (pmy_pack->pmhd != nullptr) ? pmy_pack->pmhd->u0 : pmy_pack->phydro->u0;
}

int Gravity::SourceIndex() const {
    return has_extra_density ? 0 : static_cast<int>(IDN);
}

//----------------------------------------------------------------------------------------
//! \fn Gravity::Solve()
//! \brief dispatch the Poisson solve to whichever solver was constructed
void Gravity::Solve(Driver *pdriver, int stage) {
    if (has_extra_density) {
        // total source = gas + registered density, over every cell (solvers read the
        // active zone plus one ghost layer)
        auto &u0 = (pmy_pack->pmhd != nullptr) ? pmy_pack->pmhd->u0
                                               : pmy_pack->phydro->u0;
        auto &ext = rho_extra;
        auto &tot = rho_total;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        int n3 = tot.extent_int(2), n2 = tot.extent_int(3), n1 = tot.extent_int(4);
        par_for("grav_rho_total", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            tot(m,0,k,j,i) = u0(m,IDN,k,j,i) + ext(m,0,k,j,i);
        });
    }
    if (pmgd != nullptr) {
        pmgd->Solve(pdriver, stage);
#if FFT_ENABLED
    } else if (pfft != nullptr) {
        pfft->Solve(pdriver, stage);
#endif
    }
}
} // namespace gravity
