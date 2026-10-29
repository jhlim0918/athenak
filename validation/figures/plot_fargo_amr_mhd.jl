# Figures for fargo_amr_validation.pdf, MHD with orbital advection on refined meshes
# (branch fargo-amr-mhd):
#   fargo_amr_mhd.png       -- magnetized swing amplification WITH self-gravity
#                              (inputs/tests/swing_mhd.athinput): c_y against linear theory
#                              and the deviation from the uniform twin, for static rings, a
#                              static patch and a moving ring with orbital advection, and the
#                              Phase-4 ring without it;
#   fargo_amr_mhd_faces.png -- mismatch of the two copies of the faces shared by fine and
#                              coarse blocks, Sum|Bn(coarse) - <Bn(fine)>| dA / Sum|Bn| dA
#                              (history bmis/bnrm), with and without the EMF correction of the
#                              face-field remap (gravity-free shearing wave of
#                              inputs/tests/swing_mhd_repro.athinput).
# Run directories: FARGO_AMR_RUNS (default validation/run/fargo_amr), subdirectory mhd/.
# Run from validation/: julia figures/plot_fargo_amr_mhd.jl
using CairoMakie
using Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13, Axis=(xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25))))
const C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

const HERE = @__DIR__
const VAL = dirname(HERE)
const RUNS = joinpath(get(ENV, "FARGO_AMR_RUNS", joinpath(VAL, "run", "fargo_amr")), "mhd")

function read_hst(f)
    rows = Float64[]; ncol = 0
    for ln in eachline(f)
        startswith(strip(ln), "#") && continue
        v = parse.(Float64, split(ln)); ncol = length(v); append!(rows, v)
    end
    permutedims(reshape(rows, ncol, :))
end
hst(d) = read_hst(joinpath(RUNS, d, "SwingMHD.user.hst"))

# --- Fig. 4: magnetized swing amplification WITH self-gravity (inputs/tests/swing_mhd.athinput:
# 128^2 x 16 root, multigrid gravity 4 pi G rho0 = 1.9, B0 along y, amplitude 1e-4) on
# refined meshes with orbital advection, against linear theory (swing_ode.py, 7-point
# Laplacian at the root spacing) and the uniform twins; the Phase-4 ring without orbital
# advection for comparison. Runs: mhd/swing/ (this work) and SWING_MHD_RUNS (default
# validation/run/swing_mhd/runs: uF_* uniform with, uN_* without orbital advection, smr_*/amr_*
# refined without it).
const P4 = get(ENV, "SWING_MHD_RUNS", joinpath(VAL, "run", "swing_mhd", "runs"))
const A0 = 1e-4
grav(d) = read_hst(joinpath(RUNS, "swing", d, "SwingMHD.user.hst"))
p4(d) = read_hst(joinpath(P4, d, "SwingMHD.user.hst"))
function rel(h, tw, cyw)        # (c_y - c_y,twin)/max|c_y,twin| on the twin's times
    m = tw .<= h[end, 1]
    ci = [begin
              k = clamp(searchsortedlast(h[:, 1], t), 1, size(h, 1) - 1)
              w = (t - h[k, 1]) / (h[k+1, 1] - h[k, 1])
              (1 - w) * h[k, 9] + w * h[k+1, 9]
          end for t in tw[m]]
    tw[m], (ci .- cyw[m]) ./ maximum(abs.(cyw))
end
fig = Figure(size=(1150, 760))
for (col_i, b0) in enumerate((0.5, 1.0))
    tb = @sprintf("%.1f", b0)
    th = read_hst(joinpath(P4, "uF_b" * tb, "theory.tsv"))
    a = Axis(fig[1, col_i], xlabel="t Ω", ylabel=(col_i == 1 ? "c_y / A" : ""),
             title="swing with self-gravity (4πGρ₀ = 1.9), B₀ = $tb")
    lines!(a, th[:, 1], th[:, 6], color=:black, linewidth=2.5, label="linear theory")
    u = p4("uF_b" * tb)
    lines!(a, u[:, 1], u[:, 9] ./ A0, color=:gray45, linestyle=:dot, linewidth=2,
           label="uniform 128²")
    runs = [("smr_b" * tb, true, "ring, no orbital advection (Phase 4)", :gray60, :dash),
            ("smrF_b" * tb, false, "static ring", C1, :solid),
            ("ypF_b" * tb, false, "static patch", C3, :solid)]
    b0 == 1.0 && push!(runs, ("amrF_b1.0", false, "moving ring (AMR)", C2, :solid))
    b = Axis(fig[2, col_i], xlabel="t Ω", ylabel=(col_i == 1 ? "(c_y − c_y,uniform) / max|c_y|" : ""),
             title="deviation from the uniform twin")
    for (d, phase4, lab, c, ls) in runs
        h = phase4 ? p4(d) : grav(d)
        lines!(a, h[:, 1], h[:, 9] ./ A0, color=c, linestyle=ls, linewidth=1.8, label=lab)
        tw = phase4 ? p4("uN_b" * tb) : u              # twin: same frame, uniform
        t, r = rel(h, tw[:, 1], tw[:, 9])
        lines!(b, t, r, color=c, linestyle=ls, linewidth=1.6, label=lab)
    end
    hlines!(b, [0.0], color=:black, linewidth=0.8)
    col_i == 2 && axislegend(a, position=:lt, framevisible=false, labelsize=10)
end
save(joinpath(HERE, "fargo_amr_mhd.png"), fig, px_per_unit=2)
println("wrote fargo_amr_mhd.png")

# --- shared-face consistency (gravity-free magnetized shearing wave of
# inputs/tests/swing_mhd_repro.athinput, 64^2 x 4, B0 = 1, static ring |x| < pi/4)
fig = Figure(size=(700, 420))
ax2 = Axis(fig[1, 1], yscale=log10, xlabel="t Ω", ylabel="Σ|ΔBₙ| dA / Σ|Bₙ| dA",
           title="shared fine/coarse faces: copy mismatch")
for (d, lab, col, ls) in [("f_ring_nofc", "ring, no EMF correction", C1, :dash),
                          ("f_ring", "ring, EMF correction", C1, :solid),
                          ("z3_nofc", "x₁ + x₃ jumps, no correction", C3, :dash),
                          ("z3", "x₁ + x₃ jumps, correction", C3, :solid),
                          ("amr_nofc", "moving ring (AMR), no correction", C4, :dash),
                          ("amr", "moving ring (AMR), correction", C4, :solid)]
    h = hst(d)
    r = h[2:end, 13] ./ max.(h[2:end, 14], 1e-300)
    r[r .< 1e-17] .= 1e-17
    lines!(ax2, h[2:end, 1], r, color=col, linestyle=ls, linewidth=1.6, label=lab)
end
ylims!(ax2, 1e-17, 1e8)
axislegend(ax2, position=:rt, framevisible=false, labelsize=10, nbanks=2)
save(joinpath(HERE, "fargo_amr_mhd_faces.png"), fig, px_per_unit=2)
println("wrote fargo_amr_mhd_faces.png")
