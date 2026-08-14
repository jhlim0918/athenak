# Final-density comparison for the Phase-2a swing test (doc fig, sec. 10):
# (rho - rho0) x 1e4 at the z mid-plane for uniform FFT / uniform MG / ring-SMR MG.
# Data: validation/run/bin/sw2a_*.hydro_w.*.bin (notebook sec. 12 runs).
using CairoMakie, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const HERE = @__DIR__
const VAL = dirname(HERE)
include(joinpath(VAL, "..", "scripts", "athenak_bin.jl"))
const RUN = joinpath(VAL, "run")

lastwbin(bn) = joinpath(RUN, "bin", sort(filter(f -> startswith(f, bn * ".hydro_w"),
                                               readdir(joinpath(RUN, "bin"))))[end])

# per-block z~0 slices: ((x0,x1,y0,y1), level, (rho-1)*1e4 matrix)
function slices0(bn)
    fd = read_bin(lastwbin(bn))
    nx = fd.nx_mb
    out = Tuple{NTuple{4,Float64},Int,Matrix{Float64}}[]
    for m in 1:fd.n_mbs
        g = fd.mb_geometry[m, :]
        h = (g[2] - g[1]) / nx[1]
        (g[5] <= 0 < g[6]) || continue           # one z-layer per (x,y) footprint
        k = clamp(floor(Int, -g[5] / h) + 1, 1, nx[3])
        push!(out, ((g[1], g[2], g[3], g[4]), fd.mb_logical[m, 4],
                    (fd.mb_data["dens"][m][:, :, k] .- 1.0) .* 1e4))
    end
    out
end

panels = [("uniform FFT", slices0("sw2a_unif_fft")),
          ("uniform MG", slices0("sw2a_unif_mg")),
          ("ring SMR MG", slices0("sw2a_smr"))]
cr = maximum(maximum(abs, m) for (_, ss) in panels for (_, _, m) in ss)

hms = Any[]
fig = Figure(size=(1500, 520))
for (p, (nm, ss)) in enumerate(panels)
    ax = Axis(fig[1, p]; xlabel="x", ylabel=p == 1 ? "y" : "", title=nm,
              aspect=DataAspect())
    for ((x0, x1, y0, y1), lev, m) in ss
        n1, n2 = size(m)
        xs = range(x0 + (x1-x0)/2n1, x1 - (x1-x0)/2n1, length=n1)
        ys = range(y0 + (y1-y0)/2n2, y1 - (y1-y0)/2n2, length=n2)
        push!(hms, heatmap!(ax, xs, ys, m; colorrange=(-cr, cr), colormap=:RdBu))
        p == 3 && lines!(ax, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0];
                         color=(:black, lev > 0 ? 0.35 : 0.15), linewidth=0.5)
    end
    if p == 3
        # level boundary from the data: the <refined_region> edges (pi/2 as a
        # decimal literal) sit 3e-15 inside the neighboring columns and pull them
        # in, so the fine region spans 6 of 8 block columns, NOT the nominal pi/2
        fine = [b for b in ss if b[2] > 0]
        vlines!(ax, [minimum(b[1][1] for b in fine), maximum(b[1][2] for b in fine)];
                color=(:black, 0.8), linewidth=1.5)
    end
end
Colorbar(fig[1, 4], hms[1]; label="(ρ − ρ₀) × 10⁴")

dmax = maximum(maximum(abs.(a[3] .- b[3])) for (a, b) in
               zip(slices0("sw2a_unif_fft"), slices0("sw2a_unif_mg")))
@printf("max |FFT-MG| on the uniform slice: %.1e in rho (float32 quantization scale)\n",
        dmax * 1e-4)

save(joinpath(HERE, "mg2a_swing_dens.png"), fig)
println("wrote figures/mg2a_swing_dens.png")
fig
