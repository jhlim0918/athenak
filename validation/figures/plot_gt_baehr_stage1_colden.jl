# Stage 1 of the Baehr+22 setup: gas surface density at t = 50, 100 and 150 (the first GI
# burst, the relaxation and the saturated state), and the vertical distribution of the
# stress at t = 150.
# The .npz inputs come from validation/run/gt_baehr_stage1/{colden,zprof}.py, which read
# the .bin dumps.  Usage: julia validation/figures/plot_gt_baehr_stage1_colden.jl
using CairoMakie, NPZ, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
S = joinpath(dirname(@__DIR__), "run", "gt_baehr_stage1"); Hg = 2.16878
fig = Figure(size=(1850, 540))
for (j, tag) in enumerate(("00050", "00100", "00150"))
    d = npzread(joinpath(S, "colden$tag.npz"))
    sig = d["sig"]; t = d["t"]
    x = range(d["x1min"]/Hg, d["x1max"]/Hg, length=size(sig, 2))
    y = range(d["x2min"]/Hg, d["x2max"]/Hg, length=size(sig, 1))
    ax = Axis(fig[1, j], xlabel="x / H_g", ylabel=j == 1 ? "y / H_g" : "", aspect=DataAspect(),
              title=@sprintf("Σ_g at t = %.0f Ω⁻¹   (max/mean %.2f)", t, maximum(sig)/mean(sig)))
    hm = heatmap!(ax, x, y, permutedims(sig), colormap=:magma, colorrange=(0.6, 6.3))
    j == 3 && Colorbar(fig[1, 4], hm, label="Σ_g")
end
zp = npzread(joinpath(S, "zprof150.npz"))
ax3 = Axis(fig[1, 5], xlabel="z / H_g", ylabel="⟨·⟩_xy  (normalized)",
           title="where the stress lives (t = 150)")
z = zp["z"]/Hg
lines!(ax3, z, zp["wg"]./maximum(zp["wg"]), linewidth=2, color=:crimson, label="w_grav")
lines!(ax3, z, zp["wr"]./maximum(zp["wr"]), linewidth=2, color=:royalblue, label="w_Reynolds")
lines!(ax3, z, zp["rhoz"]./maximum(zp["rhoz"]), linewidth=2, color=:black, linestyle=:dash, label="ρ")
xlims!(ax3, -5, 5); axislegend(ax3, position=:rt, framevisible=false, labelsize=12)
colsize!(fig.layout, 5, Relative(0.20))
save(joinpath(@__DIR__, "gt_baehr_stage1_colden.png"), fig, px_per_unit=2)
println("ok")
