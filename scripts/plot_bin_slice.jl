#!/usr/bin/env julia
# Plot a 2D heatmap (z≈0 slice) of one field from an AthenaK .bin dump.
# Works correctly for SMR/AMR runs (per-block data; level-1 blocks restricted
# onto the root grid). MeshBlock outlines are overlaid: gray = root level,
# green = level 1 (refined).
#
# Usage: julia scripts/plot_bin_slice.jl <file.bin> [field] [out.png]
#   field defaults to "dens"; out.png defaults to <file>_<field>.png next to
#   the input file.

using CairoMakie
include(joinpath(@__DIR__, "athenak_bin.jl"))

function main()
    fname = ARGS[1]
    field = length(ARGS) > 1 ? ARGS[2] : "dens"
    out_path = length(ARGS) > 2 ? ARGS[3] :
        replace(fname, r"\.bin$" => "") * "_$(field).png"

    fd = read_bin(fname)
    img, x1edges, x2edges = assemble_root_slice(fd, field)

    # Ωt label if this is a shearing-box run, plain t otherwise
    omega0 = try
        parse(Float64, _get_param(fd.header, "<shearing_box>", "omega0"))
    catch
        nothing
    end
    tlabel = omega0 === nothing ? "t = $(round(fd.time, digits=3))" :
        "Ωt = $(round(omega0 * fd.time, digits=3))"

    nb = (fd.Nx1 ÷ fd.nx_mb[1], fd.Nx2 ÷ fd.nx_mb[2], fd.Nx3 ÷ fd.nx_mb[3])
    title = "Mesh $(fd.Nx1)x$(fd.Nx2)x$(fd.Nx3) zones  |  " *
            "MeshBlock $(fd.nx_mb[1])x$(fd.nx_mb[2])x$(fd.nx_mb[3]) " *
            "($(nb[1])x$(nb[2])x$(nb[3]) root blocks)\n$tlabel"

    fig = Figure(size=(700, 700))
    ax = Axis(fig[1, 1], xlabel="x1", ylabel="x2", aspect=DataAspect(), title=title)
    vdev = maximum(abs.(img .- sum(img)/length(img)))
    mid = sum(img) / length(img)
    vrange = vdev < 1e-12 ? 1e-6 : vdev
    hm = heatmap!(ax, x1edges, x2edges, img, colormap=:RdBu,
                   colorrange=(mid - vrange, mid + vrange))
    Colorbar(fig[1, 2], hm, label=field)

    # overlay actual MeshBlock outlines from the file (levels colored)
    for (rect, lev) in block_outlines(fd)
        r1min, r1max, r2min, r2max = rect
        color = lev == 0 ? (:gray30, 0.6) : (:limegreen, 0.9)
        lines!(ax, [r1min, r1max, r1max, r1min, r1min],
                [r2min, r2min, r2max, r2max, r2min],
                color=color, linewidth=lev == 0 ? 0.8 : 1.5)
    end

    save(out_path, fig)
    println("Wrote $out_path")
end

main()
