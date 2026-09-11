# SC14 gravito-turbulence campaign — handoff (2026-09-10)

Reproducing Shi & Chiang (2014, ApJ 789, 34) with AthenaK's multigrid self-gravity,
in a stratified shearing box. Paper PDF: `docs/Shi_2014_ApJ_789_34.pdf`.

---

## 1. Repo state

| | |
|---|---|
| worktree here | `athenak-multigrid`, currently on **`dust-multigrid`** (separate dust project, actively developed — do not disturb) |
| SC14 work lives on | **`multigrid`**, pushed to `origin` (= `jhlim0918/athenak`) at **`7da1d18e`** |
| other worktree | `../athenak` on `test/shwave-smr` |
| cluster | TACC Vista, clone at `$HOME/athenak-multigrid` on branch `multigrid`, allocation `AST25015` |

**`multigrid` is not checked out anywhere.** To commit to it without disturbing the
dust tree, use a throwaway worktree:

```bash
git worktree add /tmp/mg-wt multigrid
# edit / copy files in /tmp/mg-wt, then commit and push from there
git -C /tmp/mg-wt push origin multigrid
git worktree remove /tmp/mg-wt
```

A plain `git push` from the dust tree pushes the wrong branch (it tracks
`upstream/feature/multigrid`); always `git push origin multigrid`.

Uncommitted and deliberately left alone: `.gitignore` (user's), `docs/` (16 MB of
paper PDFs), `scripts/cluster/archive_runs.slurm` (Vista archive job, working and
tested but never committed), `swing_G_grid_talk.png`.

---

## 2. Data — `validation/run/`

Box 64H × 64H × 12H at 256×256×48 ("standard resolution", 4 cells/H), 16³ MeshBlocks.

**The morphed ladder (use this one — it reproduces SC14 well):**

| run | β | t range | note |
|---|---|---|---|
| `gt_sc14_stndrd_b10_ng4` | 10 | 0–300 | anchor, from scratch, nghost=4 |
| `gt_sc14_stndrd_b4_morph` | 4 | 300–500 | from anchor |
| `gt_sc14_stndrd_b20_morph` | 20 | 300–450 | from anchor |
| `gt_sc14_stndrd_b40_morph` | 40 | 450–750 | from **b20** |
| `gt_sc14_stndrd_b80_morphing` | 80 | 750–1150 | from b40. **no `grav_phi` dumps** |
| `gt_sc14_stndrd_b3_amr` | 3 | 300–325 | from anchor, AMR, 101 dumps at 0.25/Ω |
| `gt_sc14_stndrd_b3_uniform_ng4` | 3 | 300–325 | same restart, uniform grid, no `grav_phi` |

**Superseded from-scratch runs** (each β started cold — worse agreement):
`gt_sc14_stndrd_b4`, `b5`, `b10`, `b40`, `b80`, `b3_morphing`; plus `gt_sc14_half`
(half-width box) and `gt_sc14_hi_b3` (hi-res).

Lineage is verifiable from output file numbering: each stage's indices continue its
parent's (anchor 0–30, b4/b20 31–…, b40 46–75, b80 76–115).

Most runs have a `gt_stress.npz` (see §3). Missing for `b80_morphing` and
`b3_uniform_ng4` — both lack `grav_phi`, so the stress extractor cannot run on them;
use their history files.

---

## 3. Tooling (all on `multigrid`)

**Extractors** (Python, run on the cluster or locally; reuse `vis/python/bin_convert.py`):
- `scripts/extract_gt_stress.py <run>` → `gt_stress.npz`: per-dump volume integrals
  (mass, stresses both weightings, `drho_rms`, z-profiles, face fluxes). Needs paired
  `hydro_w` + `grav_phi`; **pairs by header time, not file index**.
- `scripts/extract_gt_slices.py <run>` → Σ_g(x,y) and ⟨ρ⟩_y(x,z) per dump, for movies.

**Figures** (Julia/CairoMakie, in `validation/figures/`, each writes its own PNG):
`plot_sc14_alpha_vs_beta_morph.jl` (the headline α vs β, morphed ladder),
`plot_sc14_alpha_vs_beta.jl` (from-scratch version), `plot_sc14_alphaprime_vs_beta.jl`,
`plot_sc14_b10_fig2.jl` (SC14 Fig. 2 form), `plot_sc14_beta_colden.jl`,
`plot_sc14_b80_diagnosis.jl`, `plot_sc14_b3_amr.jl`, `plot_sc14_b3_amr_rhomax.jl`.

**Movies** (Julia): `scripts/movie_gt_colden_amr.jl` (Σ with refined blocks outlined),
`scripts/movie_gt_colden_compare.jl` (two runs side by side, frames matched by nearest
time), `scripts/movie_gt_xz_xy.jl` (xz + xy from `extract_gt_slices` npz).

**Cluster**: `scripts/cluster/gt_sc14_full.slurm` (Vista), `gt_sc14_beta_chain.sh`,
`archive_runs.slurm` (uncommitted).

---

## 4. Results worth not rediscovering

**Morphing matters.** The morphed ladder sits on eq. 21 (α′/eq.21 = 0.99, 1.00, 0.89,
0.87, 0.94 at β = 4, 10, 20, 40, 80) and Q is flat at 1.28–1.34 across it. δv matches
SC14 to a few percent. The from-scratch runs are worse and β = 80 fails outright.

**β = 80 has a z-boundary mass runaway, in both protocols.** The halo at |z| = 6H
lifts off the density floor, the diode boundaries become a net mass source, and mass
and energy run away (from-scratch: M ×9, E ×870). Morphing only *delays* it — dM/dt
turns positive ~63/Ω in, and SC14's trailing 200/Ω window is inside it. The plotted
β = 80 point therefore uses the run's **first** 200/Ω. Diagnosed in
`sc14_b80_diagnosis.png`. Real fix would be a taller box or a different vertical BC.

**β = 3 fragments — but only with AMR.** From the identical restart, the uniform run's
ρ_max peaks at 56 and *falls back* to 7.9 (numerical diffusion destroys the
proto-clump), while the AMR run reaches ~1000 and holds two bound clumps. The earlier
"β = 3 does not fragment at standard resolution" conclusion was a grid artifact. The
collapse is quasi-static, not free-fall: τ_growth/t_ff ≈ 56, and it is adiabatic
(t_ff ≪ t_cool), so the core heats to cs ≈ 8. λ_J/Δx floors at 6.0 — above Truelove's
4 but marginal.

**Box-averaged Q and ⟨cs⟩_ρ stop meaning anything once fragments hold a large mass
fraction.** At β = 3, midplane ⟨cs⟩_ρ reads 7.1 including fragments but 2.84 without.
Don't quote Q past the onset of fragmentation.

**AMR at standard resolution can only ever reach ONE refinement level.** The slab-open
x3 policy (`src/mesh/mesh_refinement.cpp:325`) requires
`2^(s+1)-1 ≤ lx3 ≤ nmbx3 − 2^(s+1)`; with `nmb_rootx3 = 3` that interval is empty for
all s ≥ 1. So `num_levels = 2` is the honest setting here, whatever you ask for; the
hi-res grid (6 root z-blocks) cascades fine. Also: refined regions must span the full
y extent, so one hot block drags an entire column — AMR here is much more expensive
than the clumps alone imply.

**Enabling AMR from a restart** whose input had no `<mesh_refinement>` block: pass
`-r` **and** `-i inputs/shearing_box/gravito_turb_sc14_amr.athinput` together. The
overlay's blocks get *added* (`main.cpp:280`, `FindOrAddBlock`). The overlay replaces
every parameter it contains and `Mesh::Mesh()` takes nghost from the merged set, so
`<mesh>`/`<meshblock>` must match the anchor exactly (nghost = 4 there).

**The history file beats the `.bin` dumps for time averages.** They are identical at
matched times, but dumps at 10/Ω give only 11–21 samples per window, which drives the
Reynolds estimate *negative* at β = 20 and 40. Summing the stress channels before or
after integrating is algebraically identical (checked to 1e-15) — the domain integral
is linear.

---

## 5. Open threads

1. **`rho_max` in the user history** — designed, never implemented (user stopped it).
   Recipe and the reason the naive version fails (`history.cpp` reduces every slot with
   `MPI_SUM`) are in the memory file `athenak-amr-restart-history.md`.
2. **β = 3 hi-res AMR** — the standard-res run is resolution-marginal (λ_J = 6 cells).
   Only the hi-res anchor can cascade past one AMR level.
3. **β = 80 boundary runaway** — unresolved; needs a taller box or a better vertical BC.
4. **`archive_runs.slurm`** — tested, uncommitted.
5. Vista clone is at `7da1d18e`'s parent or earlier; `git pull` there to sync.

---

## 6. Environment notes

- Plots in **Julia/CairoMakie**, not Python (user preference); Ωt time labels.
- Julia block-buffers stdout through a pipe — a script that looks hung may have
  finished. Julia's top-level `for` loops make assignments loop-local; wrap
  accumulator loops in a function.
- Vista: Grace CPUs, build with `-D Kokkos_ARCH_ARMV9_GRACE=ON`; `CMakeLists.txt`
  carries a fix for a Kokkos 4.7.2 SVE-path bug (missing `include(CheckIncludeFileCXX)`).
- Persistent memory (survives the account switch, it is filesystem-local):
  `~/.claude/projects/-Users-jaysmac-...-athenak/memory/` — see `MEMORY.md`, and
  especially `athenak-amr-restart-history.md` and `phase3-stratified-slab.md`.
