#!/usr/bin/env julia
# 3D volume rendering of one field from an AthenaK .bin dump (GLMakie, offscreen).
# Emission-absorption rendering of log10(field) in the SC14 Fig.-2 style: log color
# scale spanning the density floor to the clump peaks, box aspect true to the domain.
# SMR/AMR dumps are restricted onto the root grid first (assemble_root).
#
# Usage: julia scripts/volume_render_bin.jl <file.bin> [field] [out.png]
#   field defaults to "dens"; out.png defaults to <file>_<field>_vol.png.
# Optional env: VMIN/VMAX override the log10 color range (defaults -4 .. max);
#   CUTAWAY=1 removes the upper quadrant nearest the camera (x > 0, y < 0, z > 0),
#   SC14 Fig.-2 style, exposing the midplane (adds a _cut suffix to the default name).

using GLMakie
include(joinpath(@__DIR__, "athenak_bin.jl"))

function main()
    fname = ARGS[1]
    field = length(ARGS) > 1 ? ARGS[2] : "dens"
    cutaway = get(ENV, "CUTAWAY", "0") == "1"
    out_path = length(ARGS) > 2 ? ARGS[3] :
        replace(fname, r"\.bin$" => "") * "_$(field)_vol" * (cutaway ? "_cut" : "") *
        ".png"

    fd = read_bin(fname)
    a = assemble_root(fd, field)
    la = log10.(clamp.(a, 1e-30, Inf))

    vmin = parse(Float64, get(ENV, "VMIN", "-4.0"))
    vmax = parse(Float64, get(ENV, "VMAX", string(ceil(maximum(la); digits=1))))

    # cutaway: the upper camera-facing quadrant (x > 0, y < 0, z > 0) is simply not
    # rendered -- the remaining L-shaped region is drawn as three sub-volumes (a
    # value-carved quadrant would still absorb and hang as a dark curtain)

    lx = fd.x1max - fd.x1min
    ly = fd.x2max - fd.x2min
    lz = fd.x3max - fd.x3min

    omega0 = try
        parse(Float64, _get_param(fd.header, "<shearing_box>", "omega0"))
    catch
        NaN
    end
    tlabel = isnan(omega0) ? "t = $(round(fd.time, digits=1))" :
        "Ωt = $(round(omega0*fd.time, digits=1))"

    GLMakie.activate!()
    fig = Figure(size=(1300, 950), backgroundcolor=:white)
    ax = Axis3(fig[1, 1];
        aspect=(1.0, ly/lx, lz/lx),
        xlabel="x / H", ylabel="y / H", zlabel="z / H",
        title="log₁₀ $(field),  $(tlabel)",
        azimuth=1.2π, elevation=0.18π,
        protrusions=(60, 60, 30, 30))
    xs = range(fd.x1min, fd.x1max, length=fd.Nx1)
    ys = range(fd.x2min, fd.x2max, length=fd.Nx2)
    zs = range(fd.x3min, fd.x3max, length=fd.Nx3)
    volargs = (algorithm=:absorption, absorption=6f0,
               colormap=:turbo, colorrange=(vmin, vmax))
    subvol!(ir, jr, kr) = volume!(ax,
        xs[first(ir)] .. xs[last(ir)], ys[first(jr)] .. ys[last(jr)],
        zs[first(kr)] .. zs[last(kr)], Float32.(la[ir, jr, kr]); volargs...)

    local plt
    if cutaway
        i0 = searchsortedlast(xs, 0.0)
        j0 = searchsortedlast(ys, 0.0)
        k0 = searchsortedlast(zs, 0.0)
        plt = subvol!(1:fd.Nx1, 1:fd.Nx2, 1:k0)          # lower half, full x-y
        subvol!(1:i0, 1:fd.Nx2, k0+1:fd.Nx3)             # upper, x < 0
        subvol!(i0+1:fd.Nx1, j0+1:fd.Nx2, k0+1:fd.Nx3)   # upper, x > 0 and y > 0
    else
        plt = subvol!(1:fd.Nx1, 1:fd.Nx2, 1:fd.Nx3)
    end
    Colorbar(fig[1, 2], plt, label="log₁₀ $(field)", height=Relative(0.6))
    save(out_path, fig)
    println("wrote $out_path")
end

main()
