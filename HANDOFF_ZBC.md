# z-boundary mass leak in the stratified shearing box — handoff (2026-09-15)

**RESOLVED 2026-09-15 (dust-multigrid): option A implemented as the built-in boundary
flag `hse_outflow`** (`src/bvals/physics/hydro_bcs.cpp`, x3 faces; parser in
`src/mesh/mesh.cpp`): hydrostatic isothermal extrapolation of the edge cell under
−Ω²z, velocities copied with the normal one clamped outward, energy rebuilt, scalars
scaled; fine and coarse arrays. In the test box below (β = 80, 150/Ω): diode M ×1.158
and rising at +0.4 %/Ω⁻¹, ⟨cs⟩ ×2.04, face density 6e-2; hse_outflow M ×1.001 (peak
1.019, then −0.07 %/Ω⁻¹ outflow), ⟨cs⟩ ×1.10 saturated, face density 1e-2. At β = 10
identical to diode while the halo sits on the floor. Data `validation/run/zbc/`; write-up
in `validation/multigrid_selfgravity_validation.tex` (addendum after "why the vertical
boundary must be a diode"). Set `ix3_bc = ox3_bc = hse_outflow`; the SC14/Baehr inputs
keep diode for reproducibility of the archived runs — switch them for new campaigns.


Focused note for a fresh chat. Background on the whole SC14 campaign is in
`HANDOFF_SC14.md`; this file covers one open problem and what to do about it.

## The problem, with numbers

The `diode` x3 boundary lets mass and energy ENTER the box once the disk halo lifts
off the density floor. Every high-β SC14 run has died of it:

| run | protocol | outcome |
|---|---|---|
| `gt_sc14_stndrd_b80` (from scratch) | cold start | M ×9, E ×870, box uniformly filled by t=400 |
| `gt_sc14_stndrd_b80_morphing` | from β=40 at t=750 | dM/dt > 0 from ~63/Ω in; M ×1.97, ⟨cs⟩ 2.1→8.1 by t=1150 |
| `gt_baehr_stage1/gt_sc14_stndrd_b120` | from β=80's END | seeded already broken; M ×1.5e6, dt→3e-6, died at t=1342 |

β ≤ 40 are fine because ρ(|z|=6H) stays at the 1e-4 floor. Diagnosis figure:
`validation/figures/sc14_b80_diagnosis.png` (script `plot_sc14_b80_diagnosis.jl`).
The measured z-face mass flux (outermost-cell proxy) is inward and grows; x/y are
periodic and no cell sits on the floor after t≈100, so the z faces are the only source.

## What the boundary actually does

`src/bvals/physics/hydro_bcs.cpp:277` (inner x3) and `:316` (outer x3), acting on the
CONSERVED array `u0` (`IVZ == IM3 == 3`, so the clamp is on z-momentum):

```cpp
case BoundaryFlag::diode:
  for (int k=0; k<ng; ++k) {
    if (n==(IVZ)) u0(m,n,ke+k+1,j,i) = fmax(0.0, u0(m,n,ke,j,i));
    else          u0(m,n,ke+k+1,j,i) =           u0(m,n,ke,j,i);
  }
```

Every ghost is a copy of the last active cell (ρ, E, transverse momenta) with z-momentum
clamped outward. `outflow` is the same without the clamp (it failed faster; that is why
the input uses `diode`).

## Why it leaks

1. **The ghost is an infinite reservoir at the edge density.** The clamp constrains the
   ghost STATE, not the FLUX. With the edge cell moving inward, the face Riemann problem
   is (ρ, P, v_z<0) | (ρ, P, 0): a velocity jump across a matched contact. HLLC returns
   an intermediate velocity between v_z and 0 — negative — so the mass flux points INTO
   the box, sourced from a ghost that is regenerated every step. Mass from nothing.
2. **Hydrostatic support is removed at the face.** Copying E and ρ gives dP/dz = 0 across
   the boundary while gravity −Ω²z still pulls the edge cell inward, so the edge cell
   accelerates inward — which is exactly the condition that makes (1) fire. The BC
   manufactures the inflow it then fails to stop.
3. **Self-amplifying.** Leak rate ∝ ρ_edge. Heating → thicker disk → higher ρ_edge → more
   inflow → more gravity/compression/heating. Measured e-folding ~130/Ω at β=80.
   Injected gas carries the edge cell's rising specific energy, hence E grows faster than M.

## Fix options

- **A. Hydrostatic extrapolation in the ghosts (preferred, standard for stratified
  boxes).** Fill ghost ρ and P by integrating dP/dz = −ρΩ²z outward from the edge cell
  (isothermal in the ghost is fine: P = ρ cs²_edge), keep v_z clamped outward. Edge cell
  stays supported; ghost density falls off instead of being a flat reservoir.
  Implement as a user BC: flag `user` on ix3/ox3, enroll via `user_bcs_func`
  (`src/pgen/pgen.hpp:19,52`, `UserBoundaryFnPtr = void(*)(Mesh*)`) from
  `pgen/fluids/gravito_turb.cpp`. NB (pgen.cpp:65 warning): user BCs fill only the fine
  arrays, not the coarse arrays used for prolongation — keep refinement away from the z
  faces, which the slab-open policy already enforces.
- **B. Clamp the flux, not the state.** After the Riemann solve, zero any inward mass /
  energy / momentum flux at the physical z faces. Guaranteed one-way valve — the leak
  becomes identically zero by construction, which A only makes small. Hook point:
  `Hydro::Fluxes` / `CalculateFluxes` in `src/hydro/hydro_tasks.cpp:195`, or in the
  flux-divergence update. Cruder physically; the direct fix for the failure at hand.
- Not fixes: `vacuum` (`mesh.cpp:553`, zero ghosts) — cannot create mass but puts a hard
  rarefaction at the face and will collapse dt; `reflect` — traps the atmosphere;
  a taller box — only delays the trigger.

## How to test cheaply

Small stratified box (e.g. 32×32×24, 8³ blocks — `gravito_turb_sc14.athinput` with
`mesh/nx*` overrides; input already has `ix3_bc/ox3_bc = diode`) at β=80 from scratch,
~100/Ω. Pass/fail signals, in order of sensitivity:
1. ρ(|z|=6H) from `gt_stress.npz` `rho_z[:,0]`/`[:,-1]` (`scripts/extract_gt_stress.py`)
   — must stay near 1e-4;
2. mass (column 3 of `GravitoTurb.user.hst`) — must not trend up;
3. ⟨cs⟩_ρ = col 4 / col 3 — must saturate.
Then a real check at 256×256×48: re-run β=80 morphed from `gt_sc14_stndrd_b40_morph`
(t=750 restart) and compare with `gt_sc14_stndrd_b80_morphing`. With a working BC the
SC14 trailing window (t=950–1150) should give α′ ≈ 0.5e-2 (eq. 21), not 0.13e-2.

## Practicalities

- Code: `multigrid` branch is the SC14 campaign branch; `dust-multigrid` carries the
  corrected, ~75× faster slab solver (`0520aeed`) and is what built β=80/β=120 on Vista.
  Fix the BC on whichever you run from, then port. The pgen `<particles>` restart-block
  bug (`e02cb8b5`) is fixed on both; poisoned restarts repair with
  `scripts/fix_rst_particles.py`.
- Cluster: TACC Vista, allocation AST25015, dust tree `$HOME/athenak-multigrid`,
  SC14 worktree `$HOME/athenak-sc14` (exists, never built). Slurm twin
  `scripts/cluster/gt_sc14_full.slurm` (`ATHENAK` overridable via `--export`).
- Data: `validation/run/gt_sc14_stndrd_*`; β=120 is misplaced under
  `validation/run/gt_baehr_stage1/` (move it up).
- Memory (filesystem-local, survives account switches):
  `~/.claude/projects/-Users-jaysmac-...-athenak/memory/` — `phase3-stratified-slab.md`
  already records "use diode, not outflow" and the original inflow runaway; update it
  with the reservoir mechanism once the fix lands.
