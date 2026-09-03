# Phase 4a acceptance: the N-species damping ladder (Krapp et al. 2024 sec. 3.1) on the
# merged dust-multigrid tree.  Left: the four mean-velocity errors against the
# fine-step RK4 reference of the ODE system versus dt (tlim = 1, so dt = 1/ncycle),
# with a dt^2 guide anchored on the coarsest err_u.  Right: the total gas+dust
# momentum change over each run (must sit at round-off) for the four ladder runs and
# the three layout variants (random TSC, random NGP, single-block lattice).
#
# After the second merge (dust/dust 1f3e1ed9) the PC2 coupling ladder (coupling=pc2,
# integrator=rk2) is overlaid as open markers.
#
# Data: validation/run/dust4a/dust_damping_ladder-errs.dat and
#       validation/run/dust4a/merge2/dust_damping_ladder_pc2-errs.dat (see README.txt).
# Usage: julia validation/figures/plot_dust4a_damping.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))

const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run", "dust4a")

d = readdlm(joinpath(RUN, "dust_damping_ladder-errs.dat"), comments=true, comment_char='#')
d = d[d[:, 4] .> 0, :]                 # drop the t=0 row
lad = d[1:4, :]                        # cfl 0.4, 0.2, 0.1, 0.05
dt  = 1.0 ./ lad[:, 4]
labels = ["err_u (gas)", "err_v1 (ts=0.01)", "err_v2 (ts=0.1)", "err_v3 (ts=1.0)"]

fig = Figure(size=(1050, 430))
ax = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="dt", ylabel="mean-velocity error",
          title="damping ladder: IMEX (imex2+) and PC2 (rk2)")
for (c, lab) in enumerate(labels)
    scatterlines!(ax, dt, lad[:, 4+c], label=lab, markersize=12)
end
# PC2 ladder (second merge)
p2 = readdlm(joinpath(RUN, "merge2", "dust_damping_ladder_pc2-errs.dat"), comments=true, comment_char='#')
p2 = p2[p2[:, 4] .> 0, :][1:4, :]
dt2 = 1.0 ./ p2[:, 4]
scatterlines!(ax, dt2, p2[:, 5], color=:black, marker=:circle, markersize=12,
              strokecolor=:black, strokewidth=1.5, markercolor=:white, linestyle=:dot,
              label="err_u, PC2 (rk2)")
o2 = log.(p2[1:end-1, 5] ./ p2[2:end, 5]) ./ log.(dt2[1:end-1] ./ dt2[2:end])
text!(ax, dt[end]*1.05, lad[1, 5]*0.38,
      text=@sprintf("orders of err_u, PC2: %.2f, %.2f, %.2f", o2...), fontsize=13)
ref = lad[1, 5] .* (dt ./ dt[1]).^2
lines!(ax, dt, ref, color=:black, linestyle=:dash, label="dt^2 guide")
orders = log.(lad[1:end-1, 5] ./ lad[2:end, 5]) ./ log.(dt[1:end-1] ./ dt[2:end])
text!(ax, dt[end]*1.05, lad[1, 5]*0.6,
      text=@sprintf("orders of err_u: %.2f, %.2f, %.2f", orders...), fontsize=13)
axislegend(ax, position=:rb, framevisible=false)

ax2 = Axis(fig[1, 2], yscale=log10, xlabel="run", ylabel="|dP_total| (gas+dust)",
           title="momentum conservation: round-off",
           xticks=(1:7, ["cfl .4", "cfl .2", "cfl .1", "cfl .05", "rand TSC", "rand NGP", "1-block"]),
           xticklabelrotation=pi/6)
scatter!(ax2, 1:7, max.(d[1:7, 9], 1e-18), markersize=14)
hlines!(ax2, [1e-13], color=:gray, linestyle=:dot)
text!(ax2, 0.7, 1.4e-13, text="acceptance 1e-13", fontsize=12, color=:gray)
ylims!(ax2, 1e-17, 1e-12)
save(joinpath(HERE, "dust4a_damping.png"), fig, px_per_unit=2)
println("wrote dust4a_damping.png; err_u = ", lad[:, 5], "; orders = ", orders)
fig
