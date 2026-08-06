#!/usr/bin/env julia
# Plot volume-averaged vx and vy vs time (in orbits) from AthenaK legacy-VTK output.
#
# Usage: julia scripts/plot_vxvy.jl <run_dir>
# Reads <run_dir>/vtk/*.vtk, writes <run_dir>/vx_vy_vs_time.png.
# vx on the left axis, vy on the right axis, time divided by 2*pi (orbits).

using CairoMakie

function read_vtk_scalar(filename, field)
    data = read(filename)
    header_end = findfirst(codeunits("CELL_DATA"), data)[1] - 1
    header = String(data[1:header_end])
    dims = Int[]
    t = 0.0
    for line in split(header, "\n")
        if startswith(line, "DIMENSIONS")
            dims = parse.(Int, split(line)[2:end])
        end
        if occursin("time=", line)
            m = match(r"time=\s*([\-0-9.eE+]+)", line)
            t = parse(Float64, m.captures[1])
        end
    end
    nx, ny, nz = max.(dims .- 1, 1)
    ncells = nx * ny * nz
    marker = codeunits("SCALARS $field float")
    idx = findfirst(marker, data)
    idx === nothing && error("field '$field' not found in $filename")
    nl1 = findnext(==(0x0a), data, idx[end])
    nl2 = findnext(==(0x0a), data, nl1 + 1)
    raw = data[nl2+1:nl2+ncells*4]
    arr = Float64.(ntoh.(reinterpret(Float32, raw)))
    return arr, t
end

function main()
    run_dir = rstrip(ARGS[1], '/')
    vtk_files = sort(filter(f -> endswith(f, ".vtk"), readdir(joinpath(run_dir, "vtk"), join=true)))

    times = Float64[]
    vx = Float64[]
    vy = Float64[]
    for fn in vtk_files
        arrx, t = read_vtk_scalar(fn, "velx")
        arry, _ = read_vtk_scalar(fn, "vely")
        push!(times, t)
        push!(vx, sum(arrx) / length(arrx))
        push!(vy, sum(arry) / length(arry))
    end
    orbits = times ./ (2pi)

    fig = Figure(size=(750, 450))
    ax1 = Axis(fig[1, 1], xlabel="t / P (orbits)", ylabel="vx",
               ylabelcolor=:royalblue, yticklabelcolor=:royalblue,
               xminorticksvisible=true, xminorgridvisible=true,
               xminorticks=IntervalsBetween(5))
    ax2 = Axis(fig[1, 1], ylabel="vy", yaxisposition=:right,
               ylabelcolor=:firebrick, yticklabelcolor=:firebrick)
    hidespines!(ax2)
    hidexdecorations!(ax2)
    linkxaxes!(ax1, ax2)

    l1 = lines!(ax1, orbits, vx, color=:royalblue, linewidth=2, label="vx")
    l2 = lines!(ax2, orbits, vy, color=:firebrick, linewidth=2, label="vy")
    ax1.title = "Volume-averaged vx, vy vs time"
    axislegend(ax1, [l1, l2], ["vx", "vy"], position=:rt)

    out_path = joinpath(run_dir, "vx_vy_vs_time.png")
    save(out_path, fig)
    println("Wrote $out_path ($(length(times)) points)")
end

main()
