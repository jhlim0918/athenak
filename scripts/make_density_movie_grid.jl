#!/usr/bin/env julia
# 2x2 grid mp4 movie of a scalar field from four AthenaK runs (z≈0 slice).
#
# Usage: julia scripts/make_density_movie_grid.jl <dir1> <label1> <dir2> <label2> \
#                                                 <dir3> <label3> <dir4> <label4> [field]
# Panels fill row-major: (1,1) (1,2) (2,1) (2,2). All runs must share the same
# output cadence; the movie length is the shortest run. A shared color scale is
# used so all panels are directly comparable. MeshBlock boundaries (gray) and
# any <refined_regionN> outlines (green) are overlaid per panel.
# Writes <field>_grid2x2.mp4 into the common parent directory (runs/).

using CairoMakie
using Statistics: quantile, median, mean

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

function load_run(run_dir, label, field)
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
    return RunData(label, params, frames, times)
end

function setup_panel!(fig, row, col, run::RunData, frame_obs, vmin, vmax)
    mesh, mb = run.params["mesh"], run.params["meshblock"]
    nx1, nx2 = parse(Int, mesh["nx1"]), parse(Int, mesh["nx2"])
    mb1, mb2 = parse(Int, mb["nx1"]), parse(Int, mb["nx2"])
    x1min, x1max = parse(Float64, mesh["x1min"]), parse(Float64, mesh["x1max"])
    x2min, x2max = parse(Float64, mesh["x2min"]), parse(Float64, mesh["x2max"])
    nb1, nb2 = nx1 ÷ mb1, nx2 ÷ mb2

    title = "$(run.label)\nMesh $(nx1)x$(nx2), MeshBlock $(mb1)x$(mb2) ($(nb1)x$(nb2) blocks)"
    ax = Axis(fig[row, col], xlabel="x1", ylabel="x2", aspect=DataAspect(),
              title=title, titlesize=11, xlabelsize=11, ylabelsize=11,
              xticklabelsize=10, yticklabelsize=10)
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
                color=:limegreen, linewidth=2.0)
        text!(ax, r1min, r2max, text="level $(sec["level"])",
              color=:limegreen, fontsize=10, align=(:left, :top), offset=(3, -2))
    end
    return hm
end

function main()
    if length(ARGS) < 8
        error("usage: make_density_movie_grid.jl <dir1> <label1> ... <dir4> <label4> [field]")
    end
    field = length(ARGS) > 8 ? ARGS[9] : "dens"
    runs = [load_run(rstrip(ARGS[2i-1], '/'), ARGS[2i], field) for i in 1:4]
    nframes = minimum(length(r.frames) for r in runs)
    omega0 = parse(Float64, runs[1].params["shearing_box"]["omega0"])

    # Robust shared color scale: median across runs of each run's 99.5th-percentile
    # |deviation from the mean|, symmetric about the mean. A run with far larger
    # errors than the others (e.g. a corrupted case) saturates rather than
    # flattening the healthy panels.
    mid = mean(runs[1].frames[1])
    devs = [quantile(vec(abs.(reduce(vcat, [vec(f) for f in r.frames[1:nframes]]) .- mid)),
                     0.995) for r in runs]
    halfw = max(median(devs), 1e-10)
    vmin, vmax = mid - halfw, mid + halfw

    fig = Figure(size=(1050, 1120))
    frame_obs = [Observable(r.frames[1]) for r in runs]
    hm = setup_panel!(fig, 1, 1, runs[1], frame_obs[1], vmin, vmax)
    setup_panel!(fig, 1, 2, runs[2], frame_obs[2], vmin, vmax)
    setup_panel!(fig, 2, 1, runs[3], frame_obs[3], vmin, vmax)
    setup_panel!(fig, 2, 2, runs[4], frame_obs[4], vmin, vmax)
    Colorbar(fig[1:2, 3], hm, label=field)
    suptitle = Observable("Ωt = 0.0")
    Label(fig[0, 1:3], suptitle, fontsize=16, font=:bold)

    out_path = joinpath(dirname(rstrip(ARGS[1], '/')), "$(field)_grid2x2.mp4")
    record(fig, out_path, 1:nframes; framerate=15) do i
        for (obs, r) in zip(frame_obs, runs)
            obs[] = r.frames[i]
        end
        suptitle[] = "Ωt = $(round(omega0 * runs[1].times[i], digits=2))"
    end
    println("Wrote $out_path ($nframes frames)")
end

main()
