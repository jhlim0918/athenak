#!/usr/bin/env julia
# Side-by-side gravito-turbulence movie from the .npz written by
# scripts/extract_gt_slices.py:
#
#   left   Sigma_g(x,y) = int rho dz          (xy plane, log10 Sigma/Sigma0)
#   right  <rho_g>_y(x,z)                     (xz plane, log10 rho/rho0)
#
# The npz is read through PyCall/numpy, so no extra Julia package is needed.
#
# Usage: julia scripts/movie_gt_xz_xy.jl <slices.npz> [out.mp4]
# Optional env:
#   LAYOUT=row|col   side by side (default) or stacked with a shared x axis.
#                    The box is 64H wide and only 12H tall, so at true aspect the
#                    xz strip is 1/5 the height of the xy square: in row layout the
#                    z axis is stretched by ZSTRETCH (default 2.5, noted on the
#                    panel) to keep it legible.  LAYOUT=col wastes no space and
#                    shares the x axis -- worth a look if the strip still feels small.
#   ZSTRETCH=2.5     vertical exaggeration of the xz panel (1 = true aspect)
#   SIGMA0=2.0  RHO0=1.0            normalizations (SC14: Sigma0 = 2 rho0 H)
#   SVMIN/SVMAX  RVMIN/RVMAX        color ranges; default to the 0.2/99.8
#                                   percentiles over ALL frames (fixed, so the
#                                   movie does not breathe)
#   FPS=15  PX=1.5                  framerate and supersampling
using CairoMakie, PyCall, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=16))

npz  = ARGS[1]
out  = length(ARGS) > 1 ? ARGS[2] : replace(npz, r"\.npz$" => "") * "_xy_xz.mp4"
lay  = get(ENV, "LAYOUT", "row")
sig0 = parse(Float64, get(ENV, "SIGMA0", "2.0"))
rho0 = parse(Float64, get(ENV, "RHO0", "1.0"))
fps  = parse(Int,     get(ENV, "FPS", "15"))
zstr = parse(Float64, get(ENV, "ZSTRETCH", get(ENV, "LAYOUT", "row") == "row" ? "2.5" : "1.0"))
px   = parse(Float64, get(ENV, "PX", "1.5"))

np = pyimport("numpy")
d  = np.load(npz, allow_pickle=true)
t   = convert(Vector{Float64}, get(d, "t"))
x1  = convert(Vector{Float64}, get(d, "x1"))
x2  = convert(Vector{Float64}, get(d, "x2"))
x3  = convert(Vector{Float64}, get(d, "x3"))
sig = convert(Array{Float32,3}, get(d, "sigma"))   # (nt, nx1, nx2)
rxz = convert(Array{Float32,3}, get(d, "rho_xz"))  # (nt, nx1, nx3)
om0 = convert(Array{Float64,0}, get(d, "omega0"))[]

lsig = log10.(clamp.(sig ./ sig0, 1e-10, Inf))
lrxz = log10.(clamp.(rxz ./ rho0, 1e-10, Inf))
pct(a, p) = quantile(vec(a), p)
svmin = parse(Float64, get(ENV, "SVMIN", string(round(pct(lsig, 0.002), digits=2))))
svmax = parse(Float64, get(ENV, "SVMAX", string(round(pct(lsig, 0.998), digits=2))))
rvmin = parse(Float64, get(ENV, "RVMIN", string(round(pct(lrxz, 0.002), digits=2))))
rvmax = parse(Float64, get(ENV, "RVMAX", string(round(pct(lrxz, 0.998), digits=2))))
@printf("%d frames, Ωt = %.1f .. %.1f;  Σ range %.2f..%.2f dex, ρ range %.2f..%.2f dex\n",
        length(t), om0*t[1], om0*t[end], svmin, svmax, rvmin, rvmax)

row = lay == "row"
fig = Figure(size = row ? (1500, 660) : (1000, 1040))
title_obs = Observable(@sprintf("Ωt = %.1f", om0*t[1]))
Label(fig[0, 1:(row ? 4 : 2)], title_obs; fontsize=24, font=:bold,
      padding=(0, 0, 4, 0))

axxy = Axis(fig[1, 1]; xlabel="x / H", ylabel="y / H", aspect=DataAspect(),
            title="Σ_g(x, y)   (vertically integrated)", titlesize=17)
zttl = zstr == 1 ? "" : @sprintf("   [z ×%.3g]", zstr)
axxz = Axis(fig[row ? 1 : 2, row ? 3 : 1]; xlabel="x / H", ylabel="z / H",
            aspect=AxisAspect((x1[end]-x1[1])/((x3[end]-x3[1])*zstr)),
            title="⟨ρ_g⟩_y(x, z)   (azimuthally averaged)" * zttl, titlesize=17)

xy_obs = Observable(lsig[1, :, :])
xz_obs = Observable(lrxz[1, :, :])
hm1 = heatmap!(axxy, x1, x2, xy_obs; colormap=:inferno, colorrange=(svmin, svmax))
hm2 = heatmap!(axxz, x1, x3, xz_obs; colormap=:inferno, colorrange=(rvmin, rvmax))
Colorbar(fig[1, 2], hm1; label="log₁₀ Σ_g / Σ₀", height=Relative(0.85), width=15)
Colorbar(fig[row ? 1 : 2, row ? 4 : 2], hm2; label="log₁₀ ⟨ρ_g⟩_y / ρ₀",
         height=Relative(row ? 0.30 : 0.85), width=15)
row && colgap!(fig.layout, 2, 26)

record(fig, out, eachindex(t); framerate=fps) do n
    xy_obs[]    = lsig[n, :, :]
    xz_obs[]    = lrxz[n, :, :]
    title_obs[] = @sprintf("Ωt = %.1f", om0*t[n])
end
println("wrote $out")
