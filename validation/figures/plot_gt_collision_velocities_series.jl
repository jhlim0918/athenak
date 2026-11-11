# Phase 1 of the GI dust-growth plan, time-resolved: dust collision velocities on every particle
# snapshot (t = 150-200, every 5/Omega) of the two gtb dust runs, from
# scripts/analysis/gt_collision_velocities.py:
#   <run>/collvel/pp_000NN.npz   particle-only (all 11 snapshots; c_s = <c_s>_rho at that time)
#   <run>/collvel/gas_000NN.npz  with the gas dump (slip, arm/inter-arm split; local c_s)
# runs: gtb40_dust3_Z02_Pi01 (40 cells per H, Pi = 0.1, Vista output copied back) and
#       validation/run/gtb_BR_3sp (20 cells per H, Pi = 0)
# Writes gt_collision_velocities_series.png
# Usage: julia validation/figures/plot_gt_collision_velocities_series.jl
using CairoMakie, Printf, NPZ, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const ROOT = dirname(dirname(@__DIR__))
const RUNS = [(lab="40 cells per H, Π = 0.1", dir=joinpath(ROOT, "gtb40_dust3_Z02_Pi01", "collvel"),
               phst=joinpath(ROOT, "gtb40_dust3_Z02_Pi01", "gtb40.phst"), alpha=0.0290, etavk=0.433756/2, ls=:solid, mk=:circle),
              (lab="20 cells per H, Π = 0", dir=joinpath(ROOT, "validation", "run", "gtb_BR_3sp", "collvel"),
               phst=joinpath(ROOT, "validation", "run", "gtb_BR_3sp", "gtb.phst"), alpha=0.0345, etavk=0.0, ls=:dash, mk=:rect)]
# the particles are inserted (at rest, Gaussian layer) at t = 150: drop that snapshot from the
# time averages; panel (a) still shows it (open symbols)
const TMIN = parse(Float64, get(ENV, "TMIN", "152"))
const PK = ["time", "taus", "scales", "vfrag", "dvbins", "pairs", "hist", "ffrag", "wenv", "median"]
const GK = vcat(PK, ["slip_over_cs_median", "vx_mean", "slip_vx_mean", "env_frac"])

load(dir, pre, keys) = isdir(dir) ? [npzread(joinpath(dir, f), keys) for f in sort(readdir(dir))
                                     if startswith(f, pre) && endswith(f, ".npz")] : Dict[]
R = []
for r in RUNS
    ppall = load(r.dir, "pp_", PK); gs = filter(d -> d["time"] >= TMIN, load(r.dir, "gas_", GK))
    isempty(ppall) && (println("  (skipping $(r.lab): no $(r.dir)/pp_*.npz)"); continue)
    pp = filter(d -> d["time"] >= TMIN, ppall)
    push!(R, merge(r, (pp=pp, gs=gs, ppall=ppall)))
    println("$(r.lab): $(length(pp)) particle snapshots with t >= $TMIN (of $(length(ppall))), $(length(gs)) with gas")
end
isempty(R) && error("nothing to plot")

d0 = R[1].pp[1]
taus = vec(d0["taus"]); pairs = Int.(d0["pairs"]); npair = size(pairs, 1)
edges = vec(d0["dvbins"]); xc = sqrt.(edges[1:end-1] .* edges[2:end]); vfrag = vec(d0["vfrag"])
plab(p) = @sprintf("St %g–%g", taus[pairs[p, 1]+1], taus[pairs[p, 2]+1])
pcol = Makie.wong_colors()
function hmed(h)
    c = cumsum(h); c[end] == 0 && return NaN
    xc[min(searchsortedfirst(c, 0.5*c[end]), length(xc))]
end
medof(d, si, p) = hmed(vec(sum(d["hist"][si, p, :, :], dims=1)))
fragof(d, si, p) = (w = d["wenv"][si, p, :]; [sum(w .* replace(d["ffrag"][si, p, :, j], NaN => 0.0)) / sum(w) for j in eachindex(vfrag)])
tm(x) = (mean(x), std(x))

fig = Figure(size=(1500, 1000))
Label(fig[0, 1:3], "Dust collision velocities in gravito-turbulence, averages over t = 155–200 (10 snapshots per run; pairs in the same cell, shear-frame velocities)",
      fontsize=16, font=:bold)

# (a) time series of the one-cell median
ax = Axis(fig[1, 1], xlabel="Ω t", ylabel="median |Δv| / c_s (1-cell scale)", yscale=log10,
          title="(a) one-cell median vs time", titlealign=:left, titlefont=:bold)
for r in R, p in 1:npair
    t = [d["time"] for d in r.ppall]; m = [medof(d, 1, p) for d in r.ppall]
    scatterlines!(ax, t, m, color=pcol[p], linestyle=r.ls, marker=r.mk, markersize=6, label=(r === R[1] ? plab(p) : nothing))
end
vlines!(ax, [TMIN], color=:gray50, linestyle=:dot)
text!(ax, TMIN + 0.5, 1.5e-4, text="t = 150: insertion (excluded from averages)", fontsize=9, color=:gray30)
axislegend(ax, position=:rb, framevisible=false, labelsize=10, nbanks=2)

# (b) time-averaged median vs averaging scale, with the growth module's OC rate
ax = Axis(fig[1, 2], xlabel="averaging cell (grid cells)", ylabel="median |Δv| / c_s (mean ± s.d. over snapshots)",
          yscale=log10, xticks=[1, 2, 4], title="(b) scale dependence, time-averaged", titlealign=:left, titlefont=:bold)
for r in R, p in 1:npair
    sc = vec(r.pp[1]["scales"])
    st = [tm([medof(d, si, p) for d in r.pp]) for si in eachindex(sc)]
    mu = first.(st); sd = last.(st)
    errorbars!(ax, sc, mu, sd, color=pcol[p], whiskerwidth=6)
    scatterlines!(ax, sc, mu, color=pcol[p], linestyle=r.ls, marker=r.mk, markersize=7)
end
for St in taus
    v = sqrt(3*R[1].alpha*St); hlines!(ax, [v], color=(:gray40, 0.6), linestyle=:dot)
    text!(ax, 4.1, v, text=@sprintf("√(3αSt), St=%g", St), fontsize=10, color=:gray30, align=(:left, :center))
end
xlims!(ax, 0.8, 5.8)
text!(ax, 0.02, 0.02, space=:relative, text="solid/circles: 40 cells per H, Π = 0.1   dashed/squares: 20 cells per H, Π = 0", fontsize=10)

# (c) fragmenting fraction, time-averaged with the snapshot range
ax = Axis(fig[1, 3], xscale=log10, xlabel="v_frag / c_s", ylabel="fraction of collisions with |Δv| > v_frag",
          title="(c) fragmenting fraction (1-cell, band = snapshot range)", titlealign=:left, titlefont=:bold)
for r in R, p in 1:npair
    F = reduce(hcat, [fragof(d, 1, p) for d in r.pp])
    lo = vec(minimum(F, dims=2)); hi = vec(maximum(F, dims=2)); mu = vec(mean(F, dims=2))
    band!(ax, vfrag, lo, hi, color=(pcol[p], 0.15))
    lines!(ax, vfrag, mu, color=pcol[p], linestyle=r.ls)
end
for (vf, lab) in ((0.005, "1"), (0.015, "3"), (0.05, "10"))
    vlines!(ax, [vf], color=:gray60, linestyle=:dot); text!(ax, vf*1.04, 0.5, text="≈$lab m/s", rotation=π/2, fontsize=9, color=:gray30)
end
ylims!(ax, -0.02, 1.02)

# (d) time-averaged collision-weighted PDFs, 40 cells per H, one-cell scale
r = R[1]
ax = Axis(fig[2, 1], xscale=log10, yscale=log10, xlabel="|Δv| / c_s", ylabel="collision-weighted PDF (time-averaged)",
          title="(d) $(r.lab), 1-cell scale", titlealign=:left, titlefont=:bold)
for p in 1:npair
    h = sum(vec(sum(d["hist"][1, p, :, :], dims=1)) ./ max(sum(d["hist"][1, p, :, :]), 1e-300) for d in r.pp) ./ length(r.pp)
    h = h ./ diff(log10.(edges)); m = h .> 0
    lines!(ax, xc[m], h[m], color=pcol[p], label=plab(p))
end
xlims!(ax, 1e-4, 3); ylims!(ax, 1e-4, 10)
axislegend(ax, position=:lt, framevisible=false, labelsize=10)

# (e) arm vs inter-arm (gas snapshots), mean ± s.d.
ax = Axis(fig[2, 2], ylabel="median |Δv| / c_s (1-cell, local c_s)", yscale=log10,
          xticks=(1:npair, [plab(p) for p in 1:npair]), xticklabelrotation=π/5,
          title="(e) inter-arm (ρ/⟨ρ⟩<0.7, ▼) vs arm (>1.5, ▲), gas snapshots", titlealign=:left, titlefont=:bold)
for (ri, r) in enumerate(R)
    isempty(r.gs) && continue
    for (e, mk) in ((1, :dtriangle), (3, :utriangle))
        st = [tm([d["median"][1, p, e] for d in r.gs]) for p in 1:npair]
        x = (1:npair) .+ (ri - 1.5)*0.3 .+ (e == 1 ? -0.06 : 0.06)
        errorbars!(ax, x, first.(st), last.(st), color=(ri == 1 ? :black : :gray55), whiskerwidth=5)
        scatter!(ax, x, first.(st), marker=mk, markersize=11, color=(ri == 1 ? :black : :gray55))
    end
end
text!(ax, 0.02, 0.95, space=:relative, text=join(["$(length(r.gs)) gas snapshots: $(r.lab)" for r in R], "\n"), fontsize=10, align=(:left, :top))

# (f) mean radial drift vs time (phst), 40 cells per H Pi = 0.1, against NSH
ax = Axis(fig[2, 3], xlabel="Ω t", ylabel="⟨v_x⟩ / ηv_K", title="(f) mean radial drift (phst), $(R[1].lab)",
          titlealign=:left, titlefont=:bold)
let r = R[1]
    if isfile(r.phst) && r.etavk > 0
        rows = [split(l) for l in eachline(r.phst) if !startswith(l, "#") && !isempty(strip(l))]
        t = [parse(Float64, x[1]) for x in rows]
        keep = trues(length(t)); tmin = Inf
        for i in length(t):-1:1; t[i] < tmin ? (tmin = t[i]) : (keep[i] = false); end
        for (s, col) in enumerate((17, 30, 43))
            v = [parse(Float64, x[col]) for x in rows][keep] ./ r.etavk; tt = t[keep]
            lines!(ax, tt, v, color=(pcol[s], 0.8), label=@sprintf("St %g", taus[s]))
            nsh = -2*taus[s]/(1 + taus[s]^2); hlines!(ax, [nsh], color=pcol[s], linestyle=:dash)
        end
        axislegend(ax, position=:lb, framevisible=false, labelsize=10)
        text!(ax, 0.98, 0.98, space=:relative, text="dashed: laminar NSH", fontsize=10, align=(:right, :top))
    end
end
save(joinpath(@__DIR__, "gt_collision_velocities_series.png"), fig; px_per_unit=2)
println("ok")
