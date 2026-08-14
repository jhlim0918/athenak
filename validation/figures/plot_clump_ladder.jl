# Max-density histories for the nonlinear swing ladder (doc fig, sec. 13).
# Data: validation/run/clump_*.rhomax.txt (t, max rho slice, max rho 3D), written by
# the notebook sec. 15 cells from the runs' bin snapshots.
using CairoMakie, Printf, DelimitedFiles
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))
const HERE = @__DIR__
const RUN = joinpath(dirname(HERE), "run")
runs = [("clump_g3a",  "4πG=3.0, amp=0.05", :steelblue),
        ("clump_g3b",  "4πG=3.0, amp=0.10", :seagreen),
        ("clump_g5",   "4πG=5.0, amp=0.05", :darkorange),
        ("clump_unif", "4πG=6.0, amp=0.20", :crimson)]
fig = Figure(size=(900, 560))
ax = Axis(fig[1, 1]; xlabel="t", ylabel="max ρ / ρ₀", yscale=log10,
          title="from swing to collapse: the isothermal clumping ladder (uniform 128²×16)")
for (bn, lab, c) in runs
    f = joinpath(RUN, bn * ".rhomax.txt")
    isfile(f) || continue
    m = readdlm(f)
    lines!(ax, m[:, 1], m[:, 3]; color=c, linewidth=2, label=lab)
end
hlines!(ax, [171.0]; color=(:gray, 0.7), linestyle=:dash)
text!(ax, 0.3, 171*1.2; text="Truelove ceiling at 128² (4πG=6)", color=:gray, fontsize=11)
axislegend(ax; position=:lt, framevisible=false)
save(joinpath(HERE, "clump_ladder.png"), fig)
println("wrote figures/clump_ladder.png")
fig
