//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.cpp
//! \brief implementation of Particles class constructor and assorted other functions

#include <iostream>
#include <limits>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack) {
  // check this is at least a 2D problem
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle module only works in 2D/3D" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // read number of particles per cell, and calculate number of particles this pack
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);

  // compute number of particles as real number, since ppc can be < 1
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
  // then cast to integer
  nprtcl_thispack = static_cast<int>(r_npart);
  // no transport limit until a pusher computes one (Mesh::NewTimeStep reads dtnew before
  // the first cycle, and its 2x growth rule would lock an uninitialized zero forever)
  dtnew = std::numeric_limits<float>::max();

  // select particle type
  {
    std::string ptype = pin->GetString("particles","particle_type");
    if (ptype.compare("cosmic_ray") == 0) {
      particle_type = ParticleType::cosmic_ray;
    } else if (ptype.compare("dust") == 0) {
      particle_type = ParticleType::dust;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle type = '" << ptype << "' not recognized"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // select pusher algorithm
  {
    std::string ppush = pin->GetString("particles","pusher");
    if (ppush.compare("drift") == 0) {
      pusher = ParticlesPusher::drift;
    } else if (ppush.compare("imex_dust") == 0) {
      // dust particles are pushed by the DustGasDrag module inside the IMEX stages;
      // Particles::Push is never called for this pusher
      pusher = ParticlesPusher::imex_dust;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher must be specified in <particles> block"
                <<std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // set dimensions of particle arrays. Note particles only work in 2D/3D
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particles only work in 2D/3D, but 1D problem initialized" <<std::endl;
    std::exit(EXIT_FAILURE);
  }
  switch (particle_type) {
    case ParticleType::cosmic_ray:
      {
        int ndim=4;
        if (pmy_pack->pmesh->three_d) {ndim+=2;}
        nrdata = ndim;
        nidata = 2;
        break;
      }
    case ParticleType::dust:
      {
        // dust always carries all three position/velocity components (in 2D r-z the
        // azimuthal velocity lives in IPVZ, matching gas IM3), plus the RK registers
        // (x1,v1), the recorded drag rate R_j, stopping time, and mass
        nrdata = 17;
        nidata = 3;   // PGID, PTAG, PSP (species index)
        break;
      }
    default:
      break;
  }
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);

  // AMR: split the particles of a MeshBlock that refines (see particles_amr.cpp)
  split_on_refine = pin->GetOrAddBoolean("particles","split_on_refine",false);

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
}

//----------------------------------------------------------------------------------------
// CreateParticleTags()
// Assigns tags to particles (unique integer).  Note that tracked particles are always
// those with tag numbers less than ntrack.

void Particles::CreateParticleTags(ParameterInput *pin) {
  std::string assign = pin->GetOrAddString("particles","assign_tag","index_order");

  // tags are assigned sequentially within this rank, starting at 0 with rank=0
  if (assign.compare("index_order") == 0) {
    int tagstart = 0;
    for (int n=1; n<=global_variable::my_rank; ++n) {
      tagstart += pmy_pack->pmesh->nprtcl_eachrank[n-1];
    }

    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = tagstart + p;
    });

  // tags are assigned sequentially across ranks
  } else if (assign.compare("rank_order") == 0) {
    int myrank = global_variable::my_rank;
    int nranks = global_variable::nranks;
    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = myrank + nranks*p;
    });

  // tag algorithm not recognized, so quit with error
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle tag assignment type = '" << assign << "' not recognized"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace particles

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn int Particles::RemoveDead()
//! \brief Removes every particle whose PGID is negative (set by
//! ParticlesBoundaryValues::SetNewPrtclGID for particles that left the mesh through a
//! physical face), compacting the data arrays in place and refreshing the per-rank and
//! total particle counts of the Mesh.  Collective under MPI: every rank must call it
//! each time (the count exchange is an Allgather), whether or not it removes anything.

int Particles::RemoveDead() {
  int npart = nprtcl_thispack;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  int ndead = 0;
  if (npart > 0) {
    Kokkos::parallel_reduce("prtcl_count_dead",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &n) {
      if (pi(PGID,p) < 0) {++n;}
    }, Kokkos::Sum<int>(ndead));
  }
  if (ndead > 0) {
    // survivor index by exclusive scan, gather into fresh arrays
    int nalive = npart - ndead;
    DvceArray1D<int> map("prtcl_alive_map", npart);
    Kokkos::parallel_scan("prtcl_alive_scan",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &offset, const bool final) {
      bool alive = (pi(PGID,p) >= 0);
      if (final) {map(p) = alive ? offset : -1;}
      if (alive) {++offset;}
    });
    int nrd = nrdata, nid = nidata;
    DvceArray2D<Real> new_r("prtcl_rdata", nrd, std::max(nalive, 1));
    DvceArray2D<int>  new_i("prtcl_idata", nid, std::max(nalive, 1));
    par_for("prtcl_compact", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
      int q = map(p);
      if (q < 0) return;
      for (int n=0; n<nrd; ++n) {new_r(n,q) = pr(n,p);}
      for (int n=0; n<nid; ++n) {new_i(n,q) = pi(n,p);}
    });
    prtcl_rdata = new_r;
    prtcl_idata = new_i;
    nprtcl_thispack = nalive;
  }
  // refresh the Mesh counts (an Allgather under MPI, hence collective)
  Mesh *pm = pmy_pack->pmesh;
  pm->nprtcl_thisrank = nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&(pm->nprtcl_thisrank), 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT,
                MPI_COMM_WORLD);
#else
  pm->nprtcl_eachrank[0] = pm->nprtcl_thisrank;
#endif
  pm->nprtcl_total = 0;
  for (int n=0; n<global_variable::nranks; ++n) {
    pm->nprtcl_total += pm->nprtcl_eachrank[n];
  }
  return ndead;
}

} // namespace particles
