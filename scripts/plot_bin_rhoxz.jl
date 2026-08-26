#!/usr/bin/env julia
# Azimuthally averaged density on the x-z plane: <rho>_y(x,z) from an AthenaK .bin
# dump, as a slide-quality 2D map (CairoMakie; the SC14 Fig.-6 left-panel view).
# SMR/AMR dumps are restricted onto the root grid first.  Colors show
# log10(<rho>_y/rho0) from the halo floor to the midplane.
#
# Usage: julia scripts/plot_bin_rhoxz.jl <file.bin> [out.png]
# Optional env: VMIN/VMAX override the log10 range (defaults -4 .. 0.5);
#   RHO0 normalization (default 1.0); PX supersampling (default 2).

using CairoMakie
include(joinpath(@__DIR__, "athenak_bin.jl"))

function main()
    fname = ARGS[1]
    out_path = length(ARGS) > 1 ? ARGS[2] :
        replace(fname, r"\.bin$" => "") * "_rhoxz.png"

    fd = read_bin(fname)
    rho = assemble_root(fd, "dens")
    rho0 = parse(Float64, get(ENV, "RHO0", "1.0"))
    rxz = dropdims(sum(rho, dims=2), dims=2) ./ (fd.Nx2 * rho0)   # (Nx1, Nx3)

    vmin = parse(Float64, get(ENV, "VMIN", "-4.0"))
    vmax = parse(Float64, get(ENV, "VMAX", "0.5"))
    px = parse(Float64, get(ENV, "PX", "2"))

    omega0 = try
        parse(Float64, _get_param(fd.header, "<shearing_box>", "omega0"))
    catch
        NaN
    end
    tlabel = isnan(omega0) ? "t = $(round(fd.time, digits=1))" :
        "Ωt = $(round(omega0*fd.time, digits=1))"

    xs = range(fd.x1min, fd.x1max, length=fd.Nx1)
    zs = range(fd.x3min, fd.x3max, length=fd.Nx3)

    fig = Figure(size=(1500, 460), backgroundcolor=:white, fontsize=22)
    ax = Axis(fig[1, 1]; xlabel="x / H", ylabel="z / H",
              xlabelsize=26, ylabelsize=26, title=tlabel, titlesize=30,
              aspect=DataAspect())
    hm = heatmap!(ax, xs, zs, log10.(clamp.(rxz, 1e-10, Inf));
                  colormap=:inferno, colorrange=(vmin, vmax))
    Colorbar(fig[1, 2], hm, label="log₁₀ ⟨ρ⟩ᵧ / ρ₀", height=Relative(0.85),
             labelsize=26)
    colgap!(fig.layout, 12)
    save(out_path, fig; px_per_unit=px)
    println("wrote $out_path")
end

main()
