#!/usr/bin/env julia
# Side-by-side mp4 movie of a scalar field from two AthenaK runs (z≈0 slice).
#
# Usage: julia scripts/make_density_movie_compare.jl <run_dir_left> <run_dir_right> [field]
# Both runs must have the same output cadence. A shared color scale is used so
# the two panels are directly comparable. MeshBlock boundaries (gray) and any
# <refined_regionN> outlines (green) are overlaid per panel, as in
# make_density_movie.jl. Writes <field>_compare_<left>_vs_<right>.mp4 in runs/.

using CairoMakie

function parse_athinput(path)
    sections = Dict{String,Dict{String,String}}()
    cur = ""
    for line in eachline(path)
        line = strip(split(line, "#")[1])
        isempty(line) && continue
        if startswith(line, "<")
            cur = strip(line, ['<', '>'])
            sections[cur] = Dict{String,String}()
        elseif occursin("=", line) && cur != ""
            k, v = split(line, "=", limit=2)
            sections[cur][strip(k)] = strip(v)
        end
    end
    return sections
end

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
    return reshape(arr, (nx, ny, nz)), t
end

struct RunData
    label::String
    params::Dict{String,Dict{String,String}}
    frames::Vector{Matrix{Float64}}
    times::Vector{Float64}
end

function load_run(run_dir, field)
    athinput = only(filter(f -> endswith(f, ".athinput"), readdir(run_dir, join=true)))
    params = parse_athinput(athinput)
    # HARD GUARD: legacy VTK output is silently scrambled for multilevel meshes
    # (vtk_mesh.cpp assumes a uniform mesh). Use bin output + the *_bin scripts.
    if haskey(params, "mesh_refinement")
        error("$athinput has a <mesh_refinement> block: legacy VTK output is " *
              "invalid for SMR/AMR runs. Rerun with output file_type=bin and " *
              "use scripts/make_bin_movie_compare.jl instead.")
    end
    vtk_files = sort(filter(f -> endswith(f, ".vtk"),
                            readdir(joinpath(run_dir, "vtk"), join=true)))
    frames = Matrix{Float64}[]
    times = Float64[]
    for fn in vtk_files
        arr, t = read_vtk_scalar(fn, field)
        kz = size(arr, 3) ÷ 2 + 1
        push!(frames, arr[:, :, kz])
        push!(times, t)
    end
    return RunData(basename(run_dir), params, frames, times)
end

function setup_panel!(fig, col, run::RunData, frame_obs, vmin, vmax)
    mesh, mb = run.params["mesh"], run.params["meshblock"]
    nx1, nx2 = parse(Int, mesh["nx1"]), parse(Int, mesh["nx2"])
    mb1, mb2 = parse(Int, mb["nx1"]), parse(Int, mb["nx2"])
    x1min, x1max = parse(Float64, mesh["x1min"]), parse(Float64, mesh["x1max"])
    x2min, x2max = parse(Float64, mesh["x2min"]), parse(Float64, mesh["x2max"])
    nb1, nb2 = nx1 ÷ mb1, nx2 ÷ mb2

    title = "$(run.label)\nMesh $(nx1)x$(nx2), MeshBlock $(mb1)x$(mb2) ($(nb1)x$(nb2) blocks)"
    ax = Axis(fig[1, col], xlabel="x1", ylabel="x2", aspect=DataAspect(),
              title=title, titlesize=12)
    x1edges = range(x1min, x1max, length=nx1 + 1)
    x2edges = range(x2min, x2max, length=nx2 + 1)
    hm = heatmap!(ax, x1edges, x2edges, frame_obs,
                   colormap=:RdBu, colorrange=(vmin, vmax))

    blockw1 = (x1max - x1min) / nb1
    blockw2 = (x2max - x2min) / nb2
    vlines!(ax, [x1min + i * blockw1 for i in 1:nb1-1], color=(:gray30, 0.6), linewidth=0.8)
    hlines!(ax, [x2min + j * blockw2 for j in 1:nb2-1], color=(:gray30, 0.6), linewidth=0.8)

    for (secname, sec) in run.params
        startswith(secname, "refined_region") || continue
        r1min, r1max = parse(Float64, sec["x1min"]), parse(Float64, sec["x1max"])
        r2min, r2max = parse(Float64, sec["x2min"]), parse(Float64, sec["x2max"])
        lines!(ax, [r1min, r1max, r1max, r1min, r1min],
                [r2min, r2min, r2max, r2max, r2min],
                color=:limegreen, linewidth=2.5)
        text!(ax, r1min, r2max, text="level $(sec["level"])",
              color=:limegreen, fontsize=11, align=(:left, :top), offset=(3, -2))
    end
    return hm
end

function main()
    left_dir = rstrip(ARGS[1], '/')
    right_dir = rstrip(ARGS[2], '/')
    field = length(ARGS) > 2 ? ARGS[3] : "dens"

    left = load_run(left_dir, field)
    right = load_run(right_dir, field)
    nframes = min(length(left.frames), length(right.frames))

    omega0 = parse(Float64, left.params["shearing_box"]["omega0"])

    # shared color scale across both runs and all frames
    allvals = vcat([vec(f) for f in left.frames[1:nframes]]...,
                   [vec(f) for f in right.frames[1:nframes]]...)
    vmin, vmax = extrema(allvals)
    if vmax - vmin < 1e-10
        vmin, vmax = vmin - 1e-6, vmax + 1e-6
    end

    fig = Figure(size=(1150, 620))
    lframe = Observable(left.frames[1])
    rframe = Observable(right.frames[1])
    hm = setup_panel!(fig, 1, left, lframe, vmin, vmax)
    setup_panel!(fig, 2, right, rframe, vmin, vmax)
    Colorbar(fig[1, 3], hm, label=field)
    suptitle = Observable("Ωt = 0.0")
    Label(fig[0, 1:3], suptitle, fontsize=16, font=:bold)

    out_path = joinpath(dirname(left_dir), "$(field)_compare_$(left.label)_vs_$(right.label).mp4")
    record(fig, out_path, 1:nframes; framerate=15) do i
        lframe[] = left.frames[i]
        rframe[] = right.frames[i]
        suptitle[] = "Ωt = $(round(omega0 * left.times[i], digits=2))"
    end
    println("Wrote $out_path ($nframes frames)")
end

main()
