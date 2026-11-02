# Replica of Kim, Ostriker & Stone (2002) Fig. 3b: the critical Toomre parameter Q_c for
# axisymmetric instability of the stratified, self-gravitating, magnetized disk against the
# external-gravity parameter s0. Lines: KOS02 eq. 18 with the eq. 16 equilibrium. Symbols:
# AthenaK axisymmetric runs (three Q per case, Q_c the root of omega^2 against Q^2), as in
# KOS02 (circles beta = inf, triangles beta = 10, squares beta = 1); error bars span the
# per-run Q_c = (Q^2 - omega^2)^(1/2). Lower panel: Q_c(sim)/Q_c(eq. 18).
# Data: validation/run/kos02/fig3b_{curves,points}.txt from fig3b_data.py.
# Run from validation/: julia figures/plot_kos02_fig3b.jl
using CairoMakie
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))

const HERE = @__DIR__
const DIR = joinpath(dirname(HERE), "run", "kos02")

cur = [parse.(Float64, split(l)) for l in eachline(joinpath(DIR, "fig3b_curves.txt")) if !startswith(l, "#")]
s = getindex.(cur, 1)
qinf = [parse(Float64, x) for x in split(last(filter(l -> startswith(l, "# s0=inf"), readlines(joinpath(DIR, "fig3b_curves.txt")))))[3:5]]
pts = [split(l) for l in eachline(joinpath(DIR, "fig3b_points.txt")) if !startswith(l, "#")]

betas = ["inf", "10", "1"]
labs = ["β = ∞", "β = 10", "β = 1"]
cols = ["#2a78d6", "#eb6834", "#2e9e5b"]
styles = [:solid, :dot, :dash]
marks = [:circle, :utriangle, :rect]

fig = Figure(size=(760, 760))
ax = Axis(fig[1, 1], xscale=log10, ylabel=rich("Q", subscript("c")), xticklabelsvisible=false,
          xticks=([1e-3, 1e-2, 1e-1, 1, 10, 100], ["0.001", "0.01", "0.1", "1", "10", "100"]),
          yticks=0:0.1:1)
axr = Axis(fig[2, 1], xscale=log10, xlabel="s₀", ylabel=rich("Q", subscript("c"), " (sim) / Q", subscript("c"), " (eq. 18)"), height=190,
           xticks=([1e-3, 1e-2, 1e-1, 1, 10, 100], ["0.001", "0.01", "0.1", "1", "10", "100"]))
linkxaxes!(ax, axr)
# KOS02's own simulated values at s0 = 1 (sec. 3.2) and the rigorous s0 -> infinity, beta = inf
# result of Goldreich & Lynden-Bell (1965)
scatter!(ax, fill(1.0, 3), [0.75, 0.72, 0.57], color=:black, marker=:xcross, markersize=17,
         label="KOS02 simulations, s₀ = 1")
scatter!(ax, [150.0], [0.676], color=:black, marker=:star5, markersize=14,
         label="Goldreich & Lynden-Bell 1965, s₀ → ∞")
for (i, b) in enumerate(betas)
    lines!(ax, s, getindex.(cur, i + 1), color=cols[i], linestyle=styles[i], linewidth=2,
           label=labs[i] * " (eq. 18)")
    p = filter(r -> r[2] == b, pts)
    x = parse.(Float64, getindex.(p, 1)); q = parse.(Float64, getindex.(p, 3))
    q18 = parse.(Float64, getindex.(p, 4))
    lo = parse.(Float64, getindex.(p, 6)); hi = parse.(Float64, getindex.(p, 7))
    part = getindex.(p, 8) .== "1"
    rangebars!(ax, x, lo, hi, color=cols[i], whiskerwidth=6)
    full = .!part
    scatter!(ax, x[full], q[full], color=cols[i], marker=marks[i], markersize=11,
             label=labs[i] * " (AthenaK)")
    any(part) && scatter!(ax, x[part], q[part], color=:white, strokecolor=cols[i], strokewidth=2,
                          marker=marks[i], markersize=12)
    hlines!(axr, [1.0], color=:gray60, linewidth=1)
    scatterlines!(axr, x, q ./ q18, color=cols[i], marker=marks[i], markersize=10,
                  linestyle=styles[i])
    any(part) && scatter!(axr, x[part], (q./q18)[part], color=:white, strokecolor=cols[i],
                          strokewidth=2, marker=marks[i], markersize=10)
end
for (i, q) in enumerate(qinf)
    scatter!(ax, [150.0], [q], color=cols[i], marker=:hline, markersize=16)
end
text!(ax, 150.0, 0.475, text="s₀ → ∞", align=(:center, :top), fontsize=10, color=:gray30)
xlims!(ax, 7e-4, 220); ylims!(ax, 0, 1); ylims!(axr, 0.98, 1.05)
axislegend(ax, position=:lb, framevisible=false, labelsize=11, nbanks=2)
Label(fig[0, 1], "Critical Q for axisymmetric instability (KOS02 Fig. 3b)", fontsize=16, font=:bold)
Label(fig[3, 1], "Bars at s₀ → ∞: eq. 18 limits. Open square: s₀ = 0.01, β = 1, two of its three runs stopped at t = 16 (of 44.5).",
      fontsize=11, color=:gray30)
rowgap!(fig.layout, 8)
out = joinpath(HERE, "kos02_fig3b.png")
save(out, fig, px_per_unit=2)
println("wrote ", out)
