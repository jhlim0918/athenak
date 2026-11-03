# Self-gravity solver tests: overview for the paper (2026-11-03)

This file collects every self-gravity test we have run so far, ordered as a code-tests section
of a paper on fragmentation in self-gravitating, magnetized disks.

**Sources** (all under `athenak-multigrid/validation/` unless noted):
- FFT = `fft_selfgravity_validation.tex`
- MG = `multigrid_selfgravity_validation.tex` → `multigrid_sb_validation.pdf`
- FA = `fargo_amr_validation.tex` (on the `fargo-amr-mhd` branch, worktree
  `~/athenak-worktrees/fargo-amr/validation/`) → `fargo_amr_validation.pdf`

Line numbers refer to the `.tex` files as of 2026-11-03, after the hydro Jeans section was
added to MG.

---

## 1. Test matrix

The four test families, plus the static solver tests and the nonlinear runs that support them.

| # | Test | Gravity | B field | Rotation + shear | Uniform | Static refinement | AMR | y-local patch | FARGO |
|---|---|---|---|---|---|---|---|---|---|
| T0 | Static Poisson accuracy | ✓ | – | shear-periodic BCs | ✓ | ✓ | – | – | – |
| T1 | Hydro Jeans wave | ✓ | – | – | ✓ | ✓ (1, 2 levels) | ✓ (moving window) | – | – |
| T2 | Magneto-Jeans | ✓ | ✓ | – | ✓ | ✓ (1, 2 levels) | ✓ (moving window) | – | – |
| T3 | Swing amplification | ✓ | – | ✓ | ✓ | ✓ (ring, boundary annuli) | ✓ (moving ring) | ✓ | ✓ / ✗ |
| T4 | Magnetized swing | ✓ | ✓ | ✓ | ✓ | ✓ (ring) | ✓ (moving ring) | ✓ | ✓ (refined runs need it) |
| T5 | Nonlinear swing → collapse | ✓ | ✓ / – | ✓ | ✓ | – | ✓ (density / Jeans criterion) | ✓ | ✓ |

**Parallel checks:**
- MPI: decomposition independence was checked in every family (bit-identical or ≤ 1e-11).
- GPU (one H200): the FARGO-AMR runs, T3–T5, agree with the CPU runs to ≤ 2e-12, or ≤ 7e-9 for moving meshes.

---

## 2. The tests

### T0. Static Poisson solver accuracy (supporting; a short subsection or an appendix)
- **Periodic, discrete consistency** (FFT l.325): max relative error 3.8e-14.
  - Multigrid with a triple-sine potential (Tomida & Stone 2023, §4.1; MG l.906): second order,
    2.008 → 2.000.
  - Convergence factor 0.125 per V-cycle.
  - Identical to FFT to every digit.
- **Shear-periodic boundaries** (FFT l.335–397, MG l.813–888):
  - Bit-identical to periodic at t = 0 and t = T_sh.
  - Shearing-wave potential, multigrid error: 3.2e-5 / 4.9e-6 / 9.5e-8 at 32³ / 64³ / 128³.
  - FFT shearing-wave order: 3.0 with ppmx remap, 2.0 with plm.
  - MPI 1–8 ranks identical.
- **Vertically open (slab) boundaries** (FFT l.399, MG l.2098):
  - sech² slab matches 4πGρ₀H² ln cosh(z/H) to 0.15%.
  - Multigrid against the FFT Koyama & Ostriker (2009) solver: 3.0e-13.
  - sech² slab with multigrid: 5.8e-3 / 1.5e-3 / 3.7e-4 at 32³ / 64³ / 128³.
- **Refined meshes:**
  - Binary-potential octet control with outflow boundaries and 4 SMR levels (MG l.1022):
    - second order, error ratio 3.7–3.95;
    - normalized rms 7.5e-5, against ~1.3e-4 in Tomida & Stone.
  - Sheared static ring (MG l.1096) and boundary annuli (MG l.1364):
    - bit-identical at t = 0;
    - second order away from the interface.
- **Level-jump ghost fix** (MG l.1558–1820). The force at a coarse/fine jump was ~10% off and
  did not converge before the fix.
  - Default (`mg_fc_interp = linear`): 0.43% of the peak force at N = 256, first order.
  - `mg_fc_interp = quadratic`: 0.035%, second order (orders 1.92–1.99).
  - Phase-2a interface peak, linear / quadratic:
    - linear: 3.35e-3 / 8.19e-4 / 2.03e-4;
    - quadratic: 2.64e-3 / 6.67e-4 / 1.67e-4.
- Figures: `phase_sweep.png`, `convergence.png`, `shwave_slice.png`, `slab_profile.png`,
  `mg2a_shwave.png`. The Tomida & Stone figures (`ts41_check.png`, `ts43_check.png`) exist only in
  the notebook.

### T1. Hydro Jeans wave (gravity only)
**Dedicated dispersion suite, added 2026-11-03.** Written up in MG §13.1, "The hydrodynamic Jeans
test" (l.2419–2555), with Fig. 16 and Table 1 of `multigrid_sb_validation.pdf`. It is also part
of the abstract and the Phase 4 summary there.
- Files:
  - inputs `inputs/tests/hydro_jeans{,_smr,_amr}.athinput`;
  - runner `run/hydro_jeans/hydro_jeans.py`, results in `results.{json,tsv}`;
  - figure `figures/hydro_jeans.png`, from `plot_hydro_jeans.jl`.
- Binary: clean build of `dust-multigrid` `d0d5f564`.
- **Setup:**
  - Same box, integrator and meshes as T2: hydro module, HLLC, PPM4, RK3, CFL 0.3.
  - Adiabatic γ = 5/3, c_s = 1.
  - Multigrid: FMG + 8 V-cycles.
  - Exact eigenmodes, δρ = 1e-4: a standing wave from rest if λ < λ_J, pure growth if not.
- **Theory:**
  - continuum ω² = k²c_s² − 4πGρ₀;
  - "discrete", which uses the 7-point Laplacian eigenvalue and the centred gravity gradient of
    `SrcTerms::SelfGravity`. Gravity is weaker by about (kΔx)²/12, which matters only near
    λ = λ_J.
- **Run time:** 35 runs, about 25 min on a laptop, 6 at a time.
- **Dispersion** (λ/λ_J = 0.25–3, 11 runs, 64 cells per λ):
  - error in ω or s against the continuum: 3.7e-4 to 5.9e-3;
  - against the discrete relation: 1.8e-5 to 1.6e-3;
  - |Δω²|/k² ≤ 6e-3;
  - the worst cases sit next to marginal stability (λ/λ_J = 0.95, 1.05), where ω → 0.
- **Isothermal EOS** (λ/λ_J = 0.5, 0.95, 1.5, 2): the same as adiabatic to 2–3 digits, e.g.
  5.9e-4 / 5.2e-3 / 7.2e-4 / 4.5e-4.
- **Damping** of the stable wave: γ ≈ 5e-4 at 64 cells per λ (γ/ω ≈ 1e-4).
- **Convergence** (16 / 32 / 64 / 128 cells per λ):
  - λ/λ_J = 0.5: 1.5e-2 / 3.4e-3 / 5.9e-4 / 1.1e-4, orders 2.1–2.5;
  - λ/λ_J = 2: 1.2e-2 / 2.1e-3 / 4.4e-4 / 1.2e-4, orders 1.9–2.5.
- **3-D wave along the box diagonal** (21 cells per λ): 1.3e-3 (λ/λ_J = 0.5) and 1.7e-3 (2).
- **Refined meshes** (the same meshes as T2):
  - Error in ω, λ/λ_J = 0.5 / 2:

    | mesh | λ/λ_J = 0.5 | λ/λ_J = 2 |
    |---|---|---|
    | uni64 | 5.9e-4 | 4.4e-4 |
    | uni128 | 1.1e-4 | 1.2e-4 |
    | smr1 | 4.7e-4 | 3.5e-4 |
    | smr2 | 4.7e-4 | 3.4e-4 |
    | amr1 | 4.0e-4 | 3.5e-4 |
    | amr2 | 2.5e-4 | 2.9e-4 |

  - Every refined run lies inside the uniform bracket.
  - amr2 creates and destroys 1148/1008 blocks; the 3-D moving run 2016/1792.
  - 3-D: the central block and the moving window lower the error, 1.3e-3 → 1.1e-3 / 8.6e-4 and
    1.7e-3 → 1.5e-3.
  - Interface reflection (the spurious cos kx component): ≤ 7.8e-4 of the mode in 1-D, ≤ 9e-5
    in 3-D. It is ~1e-9 on the uniform grid.
- **Earlier evidence**, still valid:
  - FFT against multigrid end to end on a stable wave, agreement to 9 digits (FFT l.418);
  - the Tomida & Stone 2023 §4.3 Jeans-wave convergence test, order 2.27–2.29, identical to FFT
    (MG l.932–967).

### T2. Magneto-Jeans instability (gravity and field, no rotation)
MG §13 (Phase 4), l.2556–2802. Inputs `inputs/tests/magneto_jeans{,_smr,_amr}.athinput`, pgen
`jeans_wave.cpp`, runners `run/magneto_jeans{,_amr}/*.py`.

**Setup:**
- Theory: (ω² − Ω_J²)(ω² − k²v_A²cos²θ) = k²v_A² sin²θ ω² (Chandrasekhar 1961, ch. XIII).
- Exact eigenmodes, δρ = 1e-4, ∇·B = 0 from a vector potential.
- Adiabatic, γ = 5/3, c_s = 1; HLLD, PPM4, RK3.
- 64 cells per λ (quasi-1-D); 3-D diagonal runs at about 21 cells per λ.

**Uniform grid (42 runs):**
- θ = 0, 30, 60, 90°; n = 0.5–2; slow and fast branches.
- Error in ω: median 6.6e-4, maximum 4.8e-3. The near-marginal 3-D case at θ = 90° is 1.4e-2.
- Magnetic support at θ = 90°, n = 1.5: unstable to stable at v_A = 1.12 c_s, as predicted.
  ω² = −39.40 / −9.78 / 39.61 / 108.76 against −39.48 / −9.87 / 39.48 / 108.57 at
  v_A = 0.5 / 1.0 / 1.5 / 2.0.
- Convergence order 2.0–2.7, 16–128 cells per λ.
- Eigenvector δB⊥/δρ: median 4e-4.
- Wave damping γ/ω ~ 1e-4 at 64 cells.

**Refined meshes (30 runs):**
- Meshes: static slabs at level 1 and level 2; a moving AMR window crossing the box twice; a
  3-D central block with oblique interfaces.
- Every refined run lies between uniform-64 and uniform-128, except the 3-D slow wave on the
  central block (5% above uniform-64).
- Interface reflection: 6e-5 to 1.3e-3 of the mode.
- ∇·B ≤ 6e-15 through up to ~2000 blocks created and destroyed.

**Finding:** AthenaK's isothermal HLLD is numerically unstable at β ≲ 1 (MG l.2689).
- Grid noise grows at s ≈ 15 even without gravity.
- Isothermal HLLE and LLF are stable, and so is adiabatic HLLD.
- So: adiabatic runs, or HLLE for isothermal runs with a strong field.

Figures: `magneto_jeans.png`, `magneto_jeans_amr.png`.

### T3. Swing amplification (gravity, rotation and shear, no field)
**Setup:**
- One leading wave (k_x0, k_y) = (−6, 1), box 2π × 2π × π/4, 128² × 16.
- Isothermal, c_s = 1, Ω = 1, q = 1.5, 4πGρ₀ = 1.9 (just below the box's marginal value of 2),
  A = 1e-4.
- PLM, HLLE, RK2, FARGO on.
- Theory: the linear ODE system (FA appendix l.1055–1260, `run/swing_mhd/swing_ode.py`) with the
  discrete Laplacian eigenvalue. Peak δ/A = −3.605 at output time t = 7.804 (continuum −3.609).
- Inputs: `inputs/tests/swing_selfgrav{,_smr,_bndry,_amr}.athinput`, plus the fargo-amr branch's
  `swing_selfgrav_smr_ypatch.athinput`; pgen `swing.cpp`.

**Uniform convergence** (FFT l.436):

| resolution | peak δ/A | error vs −3.605 |
|---|---|---|
| 64² | 2.638 | |
| 128² | 3.511 | 2.6% |
| 256² | 3.619 | 0.4% |

- Phase lock |d_sin|/|δ|max ~ 1e-13. The kx = 0 crossing is at t = 4.005 against 4.000.
- Multigrid against FFT: peaks agree to 5.7e-8 (MG l.870).
- Without FARGO: L2 1.2e-2 against 0.9e-2 with FARGO, at 5.7× the cycles (MG l.422).

**Refined meshes** (peak δ/A; uniform 128² = −3.5114, uniform 256² = −3.6188):

| mesh | peak | vs 256² | source |
|---|---|---|---|
| static interior ring | −3.5835 (−3.5844 after the ghost fix) | −0.98% | MG l.1209, FA l.266 |
| boundary annuli at the x1 faces | −3.5502 | | MG l.1396 |
| moving ring (AMR; 392 blocks created, 280 destroyed) | −3.5110 | trajectory 2.9e-2 vs uniform | MG l.1472 |
| y-local patch (x2 level jumps) | −3.5324 | −2.39% | FA l.266 |

- In every case the refined run moves toward the fine solution, the phase stays locked, and
  2-rank runs agree with serial ones.
- A gravity-off control, run alongside each case, separates the gravity error from the hydro
  error.
- Error maps against 256² (FA): mean |Δ| = 0.047 at the ring's level jumps, 0.070 at the patch's,
  in units of ρ₀A.
- **Grid in 4πG** (MG l.1823): uniform and AMR track theory at 4πG = 1.9, 2.1 and 3.0.
  - uniform / AMR / theory: −3.51 / −3.51 / −3.61; −4.46 / |4.21| / −4.52; −8.53 / −8.26 / −8.02.

Figures: `swing.png` (FFT), `mg2a_swing_traj.png`, `mg2a_swing_dens.png`, `swing_G_grid.png`,
`fargo_amr_swing.png`, `fargo_amr_mesh_hydro.png`, `fargo_amr_errmap_hydro.png`.

### T4. Magnetized swing amplification (gravity, field, rotation and shear)
MG l.2804–2993; FA l.387–749. Inputs `inputs/tests/swing_mhd{,_repro}.athinput`, runners
`run/swing_mhd/{swing_ode.py, swing_mhd.py}`.

**Setup:**
- As T3, but adiabatic (γ = 5/3, c_s = 1), HLLD, PLM, RK2.
- B₀ along y: B₀ = 0.5 (β = 4.8) and B₀ = 1.0 (β = 1.2).
- Theory: the same ODE system with the field terms (c_x, c_y).
  - Peak δ/A: +2.299 at B₀ = 0.5, −2.847 at B₀ = 1.0.
  - max|c_y|/A: 1.57 at B₀ = 0.5, 2.00 at B₀ = 1.0.

**Uniform grid** (deviation from theory for t ≤ 6, δ / c_y):

| case | δ / c_y |
|---|---|
| B₀ = 0.5 | 0.093 / 0.049 |
| B₀ = 0.5 at 256² | 0.031 / 0.016 |
| B₀ = 1.0 | 0.041 / 0.040 |

- The FARGO frame and the full-velocity frame agree to 3.5% / 2.4%.
- ∇·B ≤ 9e-15.

**Refined meshes with FARGO** (the configuration used in production; FA l.560–749).
Deviation of c_y, B₀ = 0.5 / 1.0:

| mesh | vs theory | vs 256², t ≤ 6 |
|---|---|---|
| uniform 128² | 0.049 / 0.040 | 0.033 / 0.026 |
| ring | 0.032 / 0.027 | 0.015 / 0.013 |
| moving ring (B₀ = 1 only) | 0.028 | 0.015 |
| y-local patch | 0.034 / 0.029 | 0.018 / 0.016 |

- Against the uniform twin, to t ≤ 8: c_y within 1.7–2.6%.
- max|∇·B|/|B| ≤ 7e-15.
- Shared-face mismatch ~1e-15, thanks to the remap EMF correction (1e-2 to 0.7 without it).
- The late-time amplitude loss is damping of the wound wave, and it converges at second order:
  18% → 4.1% from 128² to 256².

**Refined meshes WITHOUT FARGO: a known defect.** State it in the paper, or avoid this
configuration.
- c_y is wrong by 7–17% against the uniform twin (MG l.2804–2993,
  `run/swing_mhd_repro/README.md`).
- The error:
  - needs field + refinement + velocity shear together;
  - is independent of gravity, the Riemann solver, reconstruction and integrator;
  - is linear in the wave amplitude;
  - converges at only ~order 0.5.
- The fine/coarse EMF correction is exact. The error enters through the fine side's interface
  EMF, which is built from prolongated ghost cells. The root cause is unfixed.
- With FARGO the same reproducer gives ring vs twin c_y = 3.3%, against 22% without (FA l.404).

Figures: `swing_mhd.png` (MG), `fargo_amr_mhd.png`, `fargo_amr_mhd_faces.png`,
`fargo_amr_mesh_mhd.png`, `fargo_amr_errmap_mhd.png`.

### T5. Nonlinear swing to collapse (bridge to the science runs)
**Hydro** (MG l.1876–2003; FA l.781–870):
- Amplitude ladder at 4πG = 3–6:
  - bounces at peak ρ/ρ₀ = 1.37 and 1.73;
  - marginal at 2.70;
  - collapse (4πG = 6, A = 0.2) at peak 203.
- Jeans-criterion AMR, rings against patches:
  - both refined runs share the first crest (25.62 at t = 4.65) and both collapse at the same
    site as the 256² run;
  - patches use 8.6× fewer MeshBlock-cycles.
- FFT twin of the 256² run: d_cos agrees to 5e-8.

**MHD** (FA l.897):
- Isothermal HLLE, B₀ = 0.5, 4πG = 6.
- Rings and patches collapse at t ≈ 9.6–9.7. The uniform 128² run does not collapse.

**Caveat:** the collapse time and place are not converged with resolution.
- The refined runs collapse 1–1.3 Ω⁻¹ earlier than 256².
- In the MHD case the 256² run collapses on the x1 boundary, where refinement is not allowed.
- A 512² reference is still needed.

Figures: `clump_ladder.png`, `fargo_amr_clump{,_maps}.png`, `fargo_amr_clump_mhd{,_maps}.png`.

### Production-scale check (not a test with an analytic answer)
- The β = 3 SC14 clump AMR on h-3 (`REPORT_clumpamr_fargoamr_2026-10-29.md`).
- The y-local mesh reproduces the ring run's fragment path: collapse → disc → first core.
- It costs ~3.4× fewer core-hours.

---

## 3. Suggested structure for the paper's code-tests section

1. **Poisson solver** (T0), short, with the numbers in a table or an appendix:
   - shear-periodic exactness;
   - second-order convergence;
   - open vertical boundaries;
   - refined meshes and level-jump ghosts.
2. **Jeans wave** (T1 + T2), possibly one figure:
   - dispersion relation ω²(k/k_J, θ, v_A) against theory, hydro (θ = 0 / B = 0) and MHD branches
     together;
   - magnetic stabilization;
   - convergence;
   - the refined-mesh panel.
3. **Swing amplification** (T3 + T4), two figures:
   - (a) δ(t) and c_y(t) against linear theory for B₀ = 0, 0.5 and 1.0, with resolution
     convergence;
   - (b) refined meshes (ring, moving ring, y-local patch) against the uniform twin, with the
     gravity-off control.
4. **Nonlinear collapse** (T5), optional, or merged into the methods discussion of the refinement
   criterion.

---

## 4. What to do before these become paper figures

1. ~~A dedicated hydro Jeans dispersion test.~~ Done 2026-11-03 (T1), and written up in MG §13.1.
   It could join T2 in one paper figure: the B = 0 branch next to the MHD branches.
2. **One code version for all figures.**
   - The tests span `dust-multigrid` (T0–T3), `fargo-amr` and `fargo-amr-mhd` (T3–T5).
   - Several T3 numbers predate the level-jump ghost fix (shifts of 0.02–0.03%).
   - Regenerate everything on one tagged commit, after merging `fargo-amr-mhd` into the main line,
     and choose `mg_fc_interp` (linear or quadratic) once.
3. **One theory number and sign convention.**
   - The swing peak is quoted as 3.60, 3.605 and 3.61, with or without the minus sign.
   - Use −3.605 (discrete Laplacian, at output time) or −3.609 (continuum), and say which.
4. **The gravity-off swing control.**
   - Its 256² peak is 1.801 against theory 1.490 (21%), yet its L2 error is 4e-3.
   - Probably a peak-definition mismatch. Resolve it before it appears in a figure.
5. **State the configuration rules:**
   - FARGO for refined MHD shearing boxes;
   - `prolong_primitives = true` for adiabatic refined boxes without FARGO;
   - adiabatic HLLD, or isothermal HLLE, at β ≲ 1;
   - the 2-D x–y shearing box is not supported (it runs silently with periodic x1 boundaries).
6. **Open MHD AMR issues** that matter for production runs:
   - ∇·B rises from 1e-13 to 5e-12 at derefinement;
   - a multi-level face mismatch of 1e-3 to 7e-3 once level 2 appears.
   Both are pre-existing AthenaK behaviour, not orbital advection.
7. **Convergence of the collapse:**
   - a 512² nonlinear swing reference;
   - refinement allowed at the x1 boundary (the `fargo-amr-x1face` work in progress).
8. **Geometry for readers:**
   - The swing box is 3-D with k_z = 0, so the closure is −4πGρ₀δ/k², not the thin-disc 2πGΣ/|k|.
   - Define the stability parameter accordingly, or quote 4πGρ₀ against the marginal value
     k_y²c_s² + κ².
9. **Doc consistency:**
   - The MG document still says "MHD + FARGO cannot refine" and lists it as not validated; the FA
     work superseded this.
   - The non-FARGO MHD error is quoted as "7–17%", "10–20%" and "13–22%" in different places.
