# Quadratic level-jump ghosts (`<gravity> mg_fc_interp`): run directory

Derivation, all result tables and the conclusions: `validation/mg_fc_quadratic_notes.md`.
Only the scripts, the patched inputs and this README are tracked (force-added; `validation/run/`
is git-ignored); the logs stay local. Binaries (git-ignored): `build-before/athena_4214ca95`
(the starting commit), `build-mpi/src/athena`, `build-fft/src/athena` (kokkos-fft, slab test).

Command-line overrides can only replace existing keys, so the inputs below are copies of the
test inputs with `mg_fc_interp` (and `show_defect`) inserted.

* `ladder.sh BIN TAG NP MB [overrides]` (env INP = input, default the committed test input):
  mg_interface_wave slab ladder, MB = 4 (4^3 blocks, N = 16..256) or 8 (8^3, N = 32..256);
  `table.py TAG MB N,N,..` prints the error table with orders; `mgiw_{lin,quad}.athinput`.
* `defsum.py LOG...`: mean V-cycle contraction, floor, cycles to the floor.
* `check_formulas.py`: ghost-formula check (O(h^3) ghost errors, flux match to round-off).
* `cube/run_cube.sh BIN TAG MB INTERP "N list" [overrides]` + `cube/table_cube.py`: 3-D
  refined-cube test (`inputs/tests/mg_interface_wave_cube.athinput`); `cube/cube2lev.athinput`
  adds a nested level-2 cube.
* `shwave/run_shwave.sh BIN TAG INPUT`: Phase-2a shwave SMR ladder (32/64/128, time0 0, 0.37);
  `shwave/shwave_{lin,quad}[_def].athinput`.
* `slab/slab_{lin,quad}.athinput`: slab-open sheet SMR test (FFT build).
* `swing/swing_{ring,patch}_quad.athinput`, `swing_amr/swing_amr_{line,quad}.athinput`:
  refined and AMR swing with quadratic ghosts.
* `swing_mhd/mhd_ring_{line,quad}.athinput`: MHD ring swing with gravity (copies of
  `validation/run/fargo_amr/mhd/swing/mesh_smr.athinput` with `mg_fc_interp`).
* `slowmode/`: the thin-slab slow-convergence probes (`mgiw_lin_sm.athinput`, `rate.sh`).
