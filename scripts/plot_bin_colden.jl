#!/usr/bin/env julia
# Vertically integrated gas surface density Sigma(x,y) = int rho dz from an AthenaK
# .bin dump, as a slide-quality 2D map (CairoMakie).  SMR/AMR dumps are restricted
# onto the root grid first.  Colors show log10(Sigma/Sigma0) with Sigma0 the initial
# SC14 surface density 2*rho0*H (= 2 in code units), so 0 = unperturbed.
#
# Usage: julia scripts/plot_bin_colden.jl <file.bin> [out.png]
# Optional env: VMIN/VMAX override the log10(Sigma/Sigma0) range (defaults -1 .. +0.7);
#   SIGMA0 overrides the normalization (default 2.0); PX supersampling (default 2).

using CairoMakie
include(joinpath(@__DIR__, "athenak_bin.jl"))

function main()
    fname = ARGS[1]
    out_path = length(ARGS) > 1 ? ARGS[2] :
        replace(fname, r"\.bin$" => "") * "_colden.png"

    fd = read_bin(fname)
    rho = assemble_root(fd, "dens")
    dz = (fd.x3max - fd.x3min)/fd.Nx3
    sigma0 = parse(Float64, get(ENV, "SIGMA0", "2.0"))
    colden = dropdims(sum(rho, dims=3), dims=3) .* dz ./ sigma0   # (Nx1, Nx2)

    vmin = parse(Float64, get(ENV, "VMIN", "-1.0"))
    vmax = parse(Float64, get(ENV, "VMAX", "0.7"))
    px = parse(Float64, get(ENV, "PX", "2"))

    omega0 = try
        parse(Float64, _get_param(fd.header, "<shearing_box>", "omega0"))
    catch
        NaN
    end
    tlabel = isnan(omega0) ? "t = $(round(fd.time, digits=1))" :
        "Ωt = $(round(omega0*fd.time, digits=1))"

    xs = range(fd.x1min, fd.x1max, length=fd.Nx1)
    ys = range(fd.x2min, fd.x2max, length=fd.Nx2)

    fig = Figure(size=(950, 860), backgroundcolor=:white, fontsize=22)
    ax = Axis(fig[1, 1]; xlabel="x / H", ylabel="y / H",
              xlabelsize=26, ylabelsize=26, title=tlabel, titlesize=30,
              aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, log10.(clamp.(colden, 1e-10, Inf));
                  colormap=:inferno, colorrange=(vmin, vmax))
    Colorbar(fig[1, 2], hm, label="log₁₀ Σ / Σ₀", height=Relative(0.85),
             labelsize=26)
    colgap!(fig.layout, 12)
    save(out_path, fig; px_per_unit=px)
    println("wrote $out_path")
end

main()
