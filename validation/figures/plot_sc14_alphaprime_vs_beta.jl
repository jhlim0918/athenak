# The clean result: volume-averaged stress alpha' (SC14 eq. 20) against cooling
# time, for the four standard-resolution runs that reach a steady state.
#
#   alpha' = (2/3) <w_xy> / <gamma P>     vs     4 / [9 gamma (gamma-1) Omega t_cool]
#
# beta = 3 (only 30/Omega after morphing, burst-dominated, no steady state) and
# beta = 80 (z-boundary mass runaway) are excluded -- see sc14_alpha_vs_beta.png
# and sc14_b80_diagnosis.png for those.
#
# Filled circles are recomputed from the .bin dumps (w_xy summed per cell before
# integrating, scripts/extract_gt_stress.py); crosses are the same quantity from
# the history file, which samples 40x more finely and is the better time average.
#
# Usage: julia validation/figures/plot_sc14_alphaprime_vs_beta.jl
using CairoMakie, Printf, DelimitedFiles, Statistics, PyCall
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))

const HERE  = @__DIR__
const RUN   = joinpath(dirname(HERE), "run")
const GAMMA = 5/3
const TCORR = 10.0
eq21(b) = 4/(9*GAMMA*(GAMMA-1))/b

# beta, run directory, trailing averaging window (SC14 Table 1 convention)
const RUNS = [(4.0,  "gt_sc14_stndrd_b4",  100.0),
              (5.0,  "gt_sc14_stndrd_b5",  100.0),
              (10.0, "gt_sc14_stndrd_b10", 100.0),
              (40.0, "gt_sc14_stndrd_b40", 200.0)]

# SC14 Table 1: beta => alpha' (x 1e-2), high resolution and standard resolution
const SC14_HI  = [(4.0,10.0), (5.0,8.19), (8.0,5.10), (10.0,4.06),
                  (20.0,2.02), (40.0,1.03), (80.0,0.52), (120.0,0.31)]
const SC14_STD = [(10.0,4.24), (20.0,2.14), (40.0,1.06), (80.0,0.55)]

function hst_alphap(dir)
    f = only(filter(x -> endswith(x, ".user.hst"), readdir(joinpath(RUN, dir), join=true)))
    d = readdlm(f, comments=true, comment_char='#')
    keep = falses(size(d, 1)); tmin = Inf
    for i in size(d, 1):-1:1
        d[i, 1] < tmin && (keep[i] = true; tmin = d[i, 1])
    end
    d = d[keep, :]
    (t=d[:, 1], ab=(2/3) .* (d[:, 8] .+ d[:, 9]) ./ d[:, 10])
end

function bin_alphap(dir)
    d = pyimport("numpy").load(joinpath(RUN, dir, "gt_stress.npz"))
    g(k) = convert(Vector{Float64}, get(d, k))
    (t=g("time"), ab=(2/3) .* g("wtot") ./ g("rho_cs2"))
end

betas = Float64[]; yb = Float64[]; eb = Float64[]; nb = Int[]; yh = Float64[]
@printf("%-5s %-12s %5s %10s %10s %10s %8s\n",
        "beta", "window", "Ndump", "a'(bin)", "a'(hst)", "eq.21", "bin/eq21")
for (beta, dir, Tavg) in RUNS
    b = bin_alphap(dir); h = hst_alphap(dir)
    mb = b.t .>= (b.t[end] - Tavg)
    mh = h.t .>= (h.t[end] - Tavg)
    v  = mean(b.ab[mb]); e = std(b.ab[mb])/sqrt(count(mb))
    vh = mean(h.ab[mh])
    push!(betas, beta); push!(yb, v); push!(eb, e); push!(nb, count(mb)); push!(yh, vh)
    @printf("%-5.0f %5.0f-%-6.0f %5d %10.2f %10.2f %10.2f %8.3f\n",
            beta, b.t[end]-Tavg, b.t[end], count(mb), 100v, 100vh, 100eq21(beta), v/eq21(beta))
end

fig = Figure(size=(800, 680))
ax = Axis(fig[1, 1]; xscale=log10, yscale=log10,
          xlabel="Ω t_cool   ( = β )",
          ylabel="α′  =  (2/3) ⟨w_xy⟩ / ⟨γP⟩      (SC14 eq. 20)",
          title="Volume-averaged stress vs cooling time\nSC14 β-scan, standard resolution (256×256×48)",
          titlealign=:left, titlesize=17, titlefont=:bold,
          xticks=([4, 5, 10, 20, 40, 80, 120], ["4","5","10","20","40","80","120"]),
          yticks=([3e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1],
                  ["3×10⁻³","5×10⁻³","10⁻²","2×10⁻²","5×10⁻²","10⁻¹","2×10⁻¹"]),
          xminorticksvisible=true, yminorticksvisible=true,
          xgridcolor=(:gray, 0.18), ygridcolor=(:gray, 0.18))
xlims!(ax, 3.4, 140); ylims!(ax, 2.4e-3, 2.2e-1)

bg = 10 .^ range(log10(3.4), log10(140), length=200)
lines!(ax, bg, eq21.(bg), color=:black, linestyle=:dash, linewidth=2.2,
       label="eq. 21:  4 / [9γ(γ−1) Ω t_cool]")

xs = [s[1] for s in SC14_HI];  ys = [s[2]*1e-2 for s in SC14_HI]
lines!(ax, xs, ys, color=(:gray, 0.45), linestyle=:dot, linewidth=1.6)
scatter!(ax, xs, ys, color=:white, strokecolor=(:black, 0.75), strokewidth=1.6,
         markersize=11, label="SC14 hi-res (Table 1)")
xs2 = [s[1] for s in SC14_STD]; ys2 = [s[2]*1e-2 for s in SC14_STD]
scatter!(ax, xs2, ys2, color=:white, strokecolor=(:black, 0.5), strokewidth=1.4,
         markersize=12, marker=:rect, label="SC14 std-res (Table 1)")

lines!(ax, betas, yb, color=:crimson, linewidth=2.4)
errorbars!(ax, betas, yb, min.(eb, 0.7 .* yb), eb, color=:crimson,
           whiskerwidth=9, linewidth=1.6)
scatter!(ax, betas, yb, color=:crimson, markersize=14, strokecolor=:white,
         strokewidth=1.2, label="this work, from .bin dumps")
scatter!(ax, betas, yh, color=:crimson, marker=:xcross, markersize=12,
         label="this work, from history file")

axislegend(ax, position=:lb, framevisible=false, labelsize=12.5, rowgap=1)
Label(fig[2, 1],
      "β = 4, 5, 10, 40 — the runs that reach a steady state.  Trailing 100/Ω (200/Ω at β = 40) averages;\n" *
      "error bars are the s.e.m. over the 11–21 dumps in each window.  The dashed line has no free parameters.",
      fontsize=11.5, color=:gray30, justification=:left, lineheight=1.3,
      padding=(0, 0, 2, 4))
rowgap!(fig.layout, 4)

out = joinpath(HERE, "sc14_alphaprime_vs_beta.png")
save(out, fig; px_per_unit=2)
println("wrote $out")
