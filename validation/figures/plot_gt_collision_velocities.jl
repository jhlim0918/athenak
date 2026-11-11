# Phase 1 of the GI dust-growth plan: dust collision velocities in gravito-turbulence, from
# scripts/analysis/gt_collision_velocities.py on the gtb box at t = 200:
#   gtb40_dust3_Z02_Pi01   40 cells per H, Pi = 0.1, dust-to-gas 0.02, St = 0.01, 0.1, 1
#   gtb_BR_3sp             20 cells per H, Pi = 0,   dust-to-gas 0.04, same St
# Writes
#   gt_collision_velocities.png   (a) median |dv|/c_s vs averaging scale, per species pair, with
#                                 the growth module's Ormel & Cuzzi rate sqrt(3 alpha St) c_s;
#                                 (b-c) |dv|/c_s distributions (40 cells per H; one-cell and
#                                 two-cell scales); (d) arm vs inter-arm medians; (e) fraction of
#                                 collisions above v_frag/c_s
#   gt_dust_drift_slip.png        particle-gas slip distributions and mean radial drift vs NSH
# Usage: julia validation/figures/plot_gt_collision_velocities.jl
using CairoMakie, Printf, NPZ, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const ROOT = dirname(dirname(@__DIR__))
const KEYS = ["taus", "scales", "vfrag", "dvbins", "pairs", "median", "hist", "ffrag", "wenv",
              "slip_hist", "slip_over_cs_median", "vx_mean", "slip_vx_mean", "cs_p_median", "dx", "Hg"]
const RUNS = [(lab="40 cells per H, Π = 0.1", f=joinpath(ROOT, "gtb40_dust3_Z02_Pi01", "collision_velocities_t200.npz"),
               alpha=0.0290, etavk=0.433756/2, ls=:solid),
              (lab="20 cells per H, Π = 0", f=joinpath(ROOT, "validation", "run", "gtb_BR_3sp", "collision_velocities_t200.npz"),
               alpha=0.0345, etavk=0.0, ls=:dash)]
D = [merge(r, (d=npzread(r.f, KEYS),)) for r in RUNS if isfile(r.f)]
isempty(D) && error("no collision_velocities_t200.npz found")

taus = vec(D[1].d["taus"]); pairs = Int.(D[1].d["pairs"]); npair = size(pairs, 1)
edges = vec(D[1].d["dvbins"]); xc = sqrt.(edges[1:end-1] .* edges[2:end])
vfrag = vec(D[1].d["vfrag"])
plab(p) = @sprintf("St %g–%g", taus[pairs[p, 1]+1], taus[pairs[p, 2]+1])
pcol = Makie.wong_colors()
# weighted median from a histogram (summed over environments unless env given)
function hmed(h)
    c = cumsum(h); c[end] == 0 && return NaN
    i = searchsortedfirst(c, 0.5*c[end]); xc[min(i, length(xc))]
end
ocrate(St, alpha) = sqrt(3*alpha*St)

fig = Figure(size=(1500, 1000))
Label(fig[0, 1:3], "Dust collision velocities in gravito-turbulence (gtb box, t = 200; pairs in the same averaging cell, velocities in the shear frame)",
      fontsize=16, font=:bold)

# (a) median vs scale
ax = Axis(fig[1, 1], xlabel="averaging cell (grid cells)", ylabel="median |Δv| / c_s", yscale=log10,
          xticks=[1, 2, 4], title="(a) median collision speed vs averaging scale", titlealign=:left, titlefont=:bold)
for r in D, p in 1:npair
    sc = vec(r.d["scales"]); m = [hmed(vec(sum(r.d["hist"][si, p, :, :], dims=1))) for si in eachindex(sc)]
    scatterlines!(ax, sc, m, color=pcol[p], linestyle=r.ls, markersize=7,
                  label=(r === D[1] ? plab(p) : nothing))
end
for (s, St) in enumerate(taus)
    hlines!(ax, [ocrate(St, D[1].alpha)], color=(:gray40, 0.6), linestyle=:dot)
    text!(ax, 4.05, ocrate(St, D[1].alpha), text=@sprintf("√(3αSt), St=%g", St), fontsize=10, color=:gray30, align=(:left, :center))
end
xlims!(ax, 0.8, 5.6)
axislegend(ax, position=:lt, framevisible=false, labelsize=10, nbanks=2)
text!(ax, 0.02, 0.02, space=:relative, text="solid: 40 cells per H, Π = 0.1   dashed: 20 cells per H, Π = 0", fontsize=10)

# (b, c) PDFs at the one- and two-cell scale, 40 cells per H
r = D[1]
for (k, si) in enumerate((1, 2))
    ax = Axis(fig[1, 1+k], xscale=log10, yscale=log10, xlabel="|Δv| / c_s", ylabel="collision-weighted PDF",
              title=@sprintf("(%s) %s, %d-cell scale", 'a'+k, r.lab, Int(r.d["scales"][si])), titlealign=:left, titlefont=:bold)
    for p in 1:npair
        h = vec(sum(r.d["hist"][si, p, :, :], dims=1)); h = h ./ (sum(h) .* diff(log10.(edges)))
        m = h .> 0
        lines!(ax, xc[m], h[m], color=pcol[p], label=plab(p))
    end
    for (vf, lab) in ((0.005, "1"), (0.015, "3"), (0.05, "10"))
        vlines!(ax, [vf], color=:gray50, linestyle=:dash)
        text!(ax, vf*1.05, 1e-3, text="v_frag/c_s = $vf (≈$lab m/s)", rotation=π/2, fontsize=9, color=:gray30)
    end
    xlims!(ax, 1e-4, 3); ylims!(ax, 1e-4, 10)
    k == 1 && axislegend(ax, position=:lt, framevisible=false, labelsize=10)
end

# (d) arm vs inter-arm medians (one-cell scale)
ax = Axis(fig[2, 1], ylabel="median |Δv| / c_s (1-cell scale)", yscale=log10,
          xticks=(1:npair, [plab(p) for p in 1:npair]), xticklabelrotation=π/5,
          title="(d) inter-arm (ρ/⟨ρ⟩ < 0.7) vs arm (> 1.5)", titlealign=:left, titlefont=:bold)
for (ri, r) in enumerate(D), (e, mk, lab) in ((1, :dtriangle, "inter-arm"), (3, :utriangle, "arm"))
    m = r.d["median"][1, :, e]
    scatter!(ax, (1:npair) .+ (ri - 1.5)*0.25, m, marker=mk, markersize=12,
             color=(ri == 1 ? :black : :gray60), label="$(ri == 1 ? "40/H" : "20/H") $lab")
end
axislegend(ax, position=:lt, framevisible=false, labelsize=10)

# (e) fraction of collisions above v_frag (one-cell scale)
ax = Axis(fig[2, 2], xscale=log10, xlabel="v_frag / c_s", ylabel="fraction of collisions with |Δv| > v_frag",
          title="(e) fragmenting fraction (1-cell scale)", titlealign=:left, titlefont=:bold)
for r in D, p in 1:npair
    w = r.d["wenv"][1, p, :]; f = [sum(w .* r.d["ffrag"][1, p, :, j]) / sum(w) for j in eachindex(vfrag)]
    scatterlines!(ax, vfrag, f, color=pcol[p], linestyle=r.ls, markersize=6, label=(r === D[1] ? plab(p) : nothing))
end
ylims!(ax, -0.02, 1.02)
axislegend(ax, position=:rt, framevisible=false, labelsize=10)

# (f) scale dependence ratio: does |Δv| shrink with the averaging cell?  median(2)/median(1)
ax = Axis(fig[2, 3], ylabel="median |Δv| at 2 cells / at 1 cell", xticks=(1:npair, [plab(p) for p in 1:npair]),
          xticklabelrotation=π/5, title="(f) resolved-gradient share: ratio 2 = pure gradient, 1 = local",
          titlealign=:left, titlefont=:bold)
for (ri, r) in enumerate(D)
    rat = [hmed(vec(sum(r.d["hist"][2, p, :, :], dims=1))) / hmed(vec(sum(r.d["hist"][1, p, :, :], dims=1))) for p in 1:npair]
    scatter!(ax, (1:npair) .+ (ri - 1.5)*0.25, rat, color=(ri == 1 ? :black : :gray60), markersize=12,
             label=(ri == 1 ? "40 cells per H" : "20 cells per H"))
end
hlines!(ax, [1.0, 2.0], color=:gray60, linestyle=:dot)
axislegend(ax, position=:rt, framevisible=false, labelsize=10)
save(joinpath(@__DIR__, "gt_collision_velocities.png"), fig; px_per_unit=2)

# ---------------- drift and slip
fig = Figure(size=(1200, 450))
ax = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="|v_p − u_g| / c_s", ylabel="PDF",
          title="(a) particle–gas slip (local gas, nearest grid point)", titlealign=:left, titlefont=:bold)
for r in D, s in eachindex(taus)
    h = vec(r.d["slip_hist"][s, :]) .* 1.0; h = h ./ (sum(h) .* diff(log10.(edges))); m = h .> 0
    lines!(ax, xc[m], h[m], color=pcol[s], linestyle=r.ls, label=(r === D[1] ? @sprintf("St %g", taus[s]) : nothing))
end
xlims!(ax, 1e-4, 3); ylims!(ax, 1e-4, 10)
axislegend(ax, position=:lt, framevisible=false)
ax = Axis(fig[1, 2], xscale=log10, xlabel="St", ylabel="⟨v_x⟩ / ηv_K",
          title="(b) mean radial drift, 40 cells per H, Π = 0.1", titlealign=:left, titlefont=:bold)
r = D[1]; ev = r.etavk
St = 10 .^ range(-2.5, 0.5, length=100)
lines!(ax, St, -2 .* St ./ (1 .+ St.^2), color=:gray40, linestyle=:dash, label="laminar NSH (no back-reaction)")
scatter!(ax, taus, vec(r.d["vx_mean"]) ./ ev, color=:black, markersize=13, label="⟨v_x⟩ particles")
scatter!(ax, taus, vec(r.d["slip_vx_mean"]) ./ ev, color=:white, strokecolor=:black, strokewidth=1.5,
         markersize=13, label="⟨v_x − u_x⟩ (slip only)")
axislegend(ax, position=:lb, framevisible=false)
save(joinpath(@__DIR__, "gt_dust_drift_slip.png"), fig; px_per_unit=2)
println("ok")
