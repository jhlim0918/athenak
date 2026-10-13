# Tomida & Stone (2023) Fig. 12 panels from the AthenaK sec. 4.4 run (job 226 on h-2,
# inputs/tests/be_collapse_mhd_ts23.athinput): (a) poloidal density with velocity vectors,
# +-250 au; (b) equatorial density with velocity vectors, +-130 au; (c) poloidal density
# with velocity vectors, +-45 au (first core); (d) plasma beta with field-direction
# vectors; (e) spherical radial velocity; (f) rotational velocity (d-f +-250 au, poloidal).
# The pgen spins the cloud clockwise about +z (m_x = rho Omega y, m_y = -rho Omega x, as
# Athena++ collapse.cpp), so (f) shows -v_phi: rotation in the sense of the initial spin,
# positive (red) as in the paper.
# Colour ranges follow the paper's panels.  Velocity arrows: length proportional to speed,
# scaled per panel (the arrow spacing corresponds to the speed in the panel title);
# field arrows are unit directions.
# Input: validation/run/be_collapse_ts23/slices_t<T>.npz from
# scripts/analysis/be_collapse_slices.py (run on h-3 next to the dumps).
# Usage: julia validation/figures/plot_be_collapse_fig12.jl [slices_t6.00.npz]
using CairoMakie, NPZ, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))
R = joinpath(dirname(@__DIR__), "run", "be_collapse_ts23")
fn = length(ARGS) > 0 ? ARGS[1] : "slices_t6.00.npz"
d = npzread(joinpath(R, fn))
tyr = d["time_yr"]

cA = "#ff2bd6"                 # arrows on cividis: magenta stays legible at both ends
fig = Figure(size=(1500, 1000))

# arrows on an every-`st`th-point lattice; velocity arrows scaled to the lattice spacing
function vectors!(ax, p, hk, vk; st=20, unit=false, color=cA)
    c = d[p * "_coord"]; h = d[p * "_" * hk]; v = d[p * "_" * vk]
    idx = 1+st÷2:st:length(c)
    pts = Point2f[]; dirs = Vec2f[]
    sp = c[st+1] - c[1]
    vmax = 0.0
    for j in idx, i in idx
        isfinite(h[j, i]) && (vmax = max(vmax, hypot(h[j, i], v[j, i])))
    end
    for j in idx, i in idx
        (isfinite(h[j, i]) && isfinite(v[j, i])) || continue
        a = hypot(h[j, i], v[j, i]); a == 0 && continue
        s = unit ? 0.8sp / a : 0.9sp / vmax
        push!(pts, Point2f(c[i], c[j])); push!(dirs, Vec2f(s*h[j, i], s*v[j, i]))
    end
    arrows2d!(ax, pts, dirs, color=color, align=:center, shaftwidth=1.2, tipwidth=5,
              tiplength=5)
    return vmax
end

tk(c) = (h = maximum(c); s = h > 200 ? 100 : (h > 100 ? 50 : 20);
         collect(-s*floor(h/s - 0.5):s:s*floor(h/s - 0.5)))  # no labels at the edges
function dens!(pos, p, cr, lab, xl, yl, ttl; vec=true)
    c = d[p * "_coord"]
    ax = Axis(pos[1, 1], aspect=DataAspect(), xlabel=xl, ylabel=yl, title=ttl,
              titlealign=:left, xticks=tk(c), yticks=tk(c))
    hm = heatmap!(ax, c, c, permutedims(clamp.(d[p * "_rho"], cr...)),
                  colormap=:cividis, colorscale=log10, colorrange=cr)
    vmax = vec ? vectors!(ax, p, "vh", "vv") : 0.0
    limits!(ax, extrema(c)..., extrema(c)...)
    Colorbar(pos[1, 2], hm, label="ρ [g cm⁻³]", scale=log10)
    ax, vmax
end

c = d["large_coord"]
ax, v1 = dens!(fig[1, 1], "large", (3e-17, 1e-12), "", "x [au]", "z [au]", "(a)")
ax.title = @sprintf("(a) poloidal, arrows ≤ %.2f km/s", v1)
ax, v2 = dens!(fig[1, 2], "eq", (1e-15, 1e-11), "", "x [au]", "y [au]", "(b)")
ax.title = @sprintf("(b) equatorial, arrows ≤ %.2f km/s", v2)
ax, v3 = dens!(fig[1, 3], "core", (3e-16, 1e-10), "", "x [au]", "z [au]", "(c)")
ax.title = @sprintf("(c) first core, arrows ≤ %.2f km/s", v3)

# (d) plasma beta: diverging about beta = 1 (blue magnetically, red thermally dominated)
axd = Axis(fig[2, 1][1, 1], aspect=DataAspect(), xlabel="x [au]", ylabel="z [au]",
           title="(d) plasma β, field directions", titlealign=:left, xticks=tk(c),
           yticks=tk(c))
hb = heatmap!(axd, c, c, permutedims(clamp.(d["large_beta"], 1e-3, 1e3)),
              colormap=Reverse(:RdBu), colorscale=log10, colorrange=(1e-3, 1e3))
vectors!(axd, "large", "bh", "bv", unit=true, color=:black)
limits!(axd, extrema(c)..., extrema(c)...)
Colorbar(fig[2, 1][1, 2], hb, label="β", scale=log10)

for (k, (key, lab, ttl)) in enumerate((("large_vr", "v_r [km s⁻¹]",
                                        "(e) radial velocity"),
                                       ("large_vphi", "rotation velocity [km s⁻¹]",
                                        "(f) rotational velocity")))
    a = Axis(fig[2, k + 1][1, 1], aspect=DataAspect(), xlabel="x [au]", ylabel="z [au]",
             title=ttl, titlealign=:left, xticks=tk(c), yticks=tk(c))
    sgn = key == "large_vphi" ? -1 : 1
    h = heatmap!(a, c, c, permutedims(clamp.(sgn .* d[key], -2, 2)),
                 colormap=Reverse(:RdBu), colorrange=(-2, 2))
    limits!(a, extrema(c)..., extrema(c)...)
    Colorbar(fig[2, k + 1][1, 2], h, label=lab)
end
rhoc = maximum(filter(isfinite, d["core_rho"]))
Label(fig[0, 1:3], "AthenaK, Tomida & Stone (2023) §4.4 setup: t = " *
      @sprintf("%.0f", tyr) * " yr, ρ_c = " * @sprintf("%.1e", rhoc) * " g cm⁻³",
      font=:bold)
colgap!(fig.layout, 30)
out = replace(fn, "slices_" => "be_collapse_fig12_", ".npz" => ".png")
save(joinpath(@__DIR__, out), fig, px_per_unit=1.5)
println("wrote ", out)
