# The t = 0 velocity seed of the SC14 beta = 10 run (gt_sc14_stndrd_b10, plm; the hse and
# ppm4/wenoz twins start from the identical field): independent uniform noise of amplitude
# 0.1 cs0 per component, cell by cell, for |z| < 2H.
#   (a-c) vx, vy, vz in the midplane layer;  (d) a 8H x 8H zoom of vx;  (e) vz in x-z at y = 0
#   (f) 1D power per mode along x and y;  (g) 2D shell spectrum E(k);  (h) cumulative
#   fraction of the seeded kinetic energy below k, with the most unstable GI wavelength
# Input: python3 scripts/analysis/gt_initial_velocity_extract.py validation/run/gt_sc14_stndrd_b10 \
#            -o validation/run/gt_sc14_stndrd_b10/initial_velocity.npz
# Usage: julia validation/figures/plot_sc14_b10_initial_velocity.jl
using CairoMakie, Printf, NPZ
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const RUN = joinpath(dirname(@__DIR__), "run", "gt_sc14_stndrd_b10")
d = npzread(joinpath(RUN, "initial_velocity.npz"))
const CS0 = 2.12625
const VP = 0.1*CS0
x = range(d["x1min"], d["x1max"], length=size(d["mid_vx"], 2) + 1)[1:end-1] .+ 0.125
y = range(d["x2min"], d["x2max"], length=size(d["mid_vx"], 1) + 1)[1:end-1] .+ 0.125
zc = vec(d["zc"])
kmu = 1/CS0                       # most unstable k at Q = 1: k = pi G Sigma/cs^2 = Omega/cs
k4 = 2π/(4*0.25)                  # 4-cell wavelength
kny = π/0.25                      # Nyquist

fig = Figure(size=(1500, 1000))
Label(fig[0, 1:4], @sprintf("SC14 β = 10, t = 0: seeded velocity (uniform noise ±0.1 cs₀ = ±%.3f per cell, |z| < 2H; rms %.4f, %.4f, %.4f)",
      VP, d["rms_vx"], d["rms_vy"], d["rms_vz"]), fontsize=16, font=:bold)
cmap = Reverse(:RdBu)
hm = nothing
for (i, c) in enumerate(("vx", "vy", "vz"))
    ax = Axis(fig[1, i], aspect=DataAspect(), title=@sprintf("(%s) %s, midplane layer z = %+.3f H", 'a' + i - 1, c, zc[d["kmid"]+1]),
              titlealign=:left, titlefont=:bold, xlabel="x / H", ylabel=i == 1 ? "y / H" : "")
    global hm = heatmap!(ax, x, y, d["mid_$c"]', colormap=cmap, colorrange=(-VP, VP))
end
ax = Axis(fig[1, 4], aspect=DataAspect(), title="(d) vx, zoom 8H × 8H", titlealign=:left,
          titlefont=:bold, xlabel="x / H")
m = (x .>= -4) .& (x .<= 4); n = (y .>= -4) .& (y .<= 4)
heatmap!(ax, x[m], y[n], d["mid_vx"][n, m]', colormap=cmap, colorrange=(-VP, VP))
Colorbar(fig[1, 5], hm, label="v / (HΩ)")

axz = Axis(fig[2, 1:2], title="(e) vz in the x–z plane at y = 0", titlealign=:left, titlefont=:bold,
           xlabel="x / H", ylabel="z / H", aspect=DataAspect())
heatmap!(axz, x, zc, d["xz_vz"]', colormap=cmap, colorrange=(-VP, VP))
hlines!(axz, [-2, 2], color=:black, linestyle=:dash, linewidth=1)

sx = Axis(fig[2, 3], xscale=log10, yscale=log10, xlabel="k H", ylabel="|v̂(k)|² / N²",
          title="(f) 1D power per mode (solid: along x, dashed: along y)", titlealign=:left, titlefont=:bold)
cols = Dict("vx" => RGBf(0.80, 0.14, 0.22), "vy" => RGBf(0.17, 0.42, 0.72), "vz" => :black)
for c in ("vx", "vy", "vz")
    lines!(sx, vec(d["kx"])[2:end], vec(d["px_$c"])[2:end], color=cols[c], label=c)
    lines!(sx, vec(d["ky"])[2:end], vec(d["py_$c"])[2:end], color=(cols[c], 0.6), linestyle=:dash)
end
hlines!(sx, [d["rms_vx"]^2/256], color=:gray40, linestyle=:dot)
text!(sx, 0.12, d["rms_vx"]^2/256*1.6, text="white noise: σ²/N", fontsize=11, color=:gray30)
axislegend(sx, position=:rb, framevisible=false)
ylims!(sx, 1e-6, 1e-3)

se = Axis(fig[2, 4], xscale=log10, yscale=log10, xlabel="k H", ylabel="E(k)  and  cumulative fraction",
          title="(g) 2D shell spectrum E(k) (solid) and energy below k (dashed)", titlealign=:left, titlefont=:bold)
k = vec(d["k"]); ek = vec(d["E_k"]); cf = vec(d["cumfrac"])
lines!(se, k, ek, color=:black, label="E(k)")
lines!(se, k, cf, color=RGBf(0.80, 0.14, 0.22), linestyle=:dash, label="fraction below k")
for (kv, lab) in ((kmu, "λ_GI = 2πcs₀/Ω ≈ 13.4H"), (k4, "4 cells"), (kny, "Nyquist"))
    vlines!(se, [kv], color=:gray50, linestyle=:dot)
    text!(se, kv*1.05, 3e-6, text=lab, fontsize=10, rotation=π/2, color=:gray30)
end
i = argmin(abs.(k .- kmu))
@printf("fraction of the seeded KE at k < k_GI = %.3f/H (lambda > %.1f H): %.2e\n", kmu, 2π/kmu, cf[i])
@printf("fraction at lambda > 4 cells (k < %.2f): %.3f\n", k4, cf[argmin(abs.(k .- k4))])
text!(se, 0.11, 0.3, text=@sprintf("%.1e of the KE at λ > 13.4H", cf[i]), fontsize=11, color=RGBf(0.80, 0.14, 0.22))
ylims!(se, 1e-6, 2.0)
axislegend(se, position=:rb, framevisible=false)
save(joinpath(@__DIR__, "sc14_b10_initial_velocity.png"), fig; px_per_unit=2)
println("ok")
