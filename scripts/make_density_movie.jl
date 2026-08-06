#!/usr/bin/env julia
# Make an mp4 movie of a z≈0 slice of a scalar field from AthenaK legacy-VTK output.
#
# Usage: julia scripts/make_density_movie.jl <run_dir> [field]
# Reads <run_dir>/vtk/*.vtk and the copied *.athinput in <run_dir> for mesh
# info, and writes <run_dir>/<field>_movie.mp4. Time is labeled as Ωt (code
# units), with omega0 taken from the <shearing_box> block. The z-slice line
# in the title is omitted for purely 2D runs (nx3=1).

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

function main()
    run_dir = rstrip(ARGS[1], '/')
    field = length(ARGS) > 1 ? ARGS[2] : "dens"

    athinput = only(filter(f -> endswith(f, ".athinput"), readdir(run_dir, join=true)))
    params = parse_athinput(athinput)
    # HARD GUARD: legacy VTK output is silently scrambled for multilevel meshes
    # (vtk_mesh.cpp assumes a uniform mesh). Use bin output + the *_bin scripts.
    if haskey(params, "mesh_refinement")
        error("$athinput has a <mesh_refinement> block: legacy VTK output is " *
              "invalid for SMR/AMR runs. Rerun with output file_type=bin and " *
              "use scripts/plot_bin_slice.jl / make_bin_movie_compare.jl instead.")
    end
    mesh, mb, sbox = params["mesh"], params["meshblock"], params["shearing_box"]
    nx1, nx2, nx3 = parse.(Int, (mesh["nx1"], mesh["nx2"], mesh["nx3"]))
    mb1, mb2, mb3 = parse.(Int, (mb["nx1"], mb["nx2"], mb["nx3"]))
    x1min, x1max = parse.(Float64, (mesh["x1min"], mesh["x1max"]))
    x2min, x2max = parse.(Float64, (mesh["x2min"], mesh["x2max"]))
    x3min, x3max = parse.(Float64, (mesh["x3min"], mesh["x3max"]))
    omega0 = parse(Float64, sbox["omega0"])
    nblocks = (nx1 ÷ mb1, nx2 ÷ mb2, nx3 ÷ mb3)

    vtk_files = sort(filter(f -> endswith(f, ".vtk"), readdir(joinpath(run_dir, "vtk"), join=true)))

    frames = Matrix{Float64}[]
    times = Float64[]
    zslice_coord = 0.0
    for fn in vtk_files
        arr, t = read_vtk_scalar(fn, field)
        nz = size(arr, 3)
        kz = nz ÷ 2 + 1
        dz = (x3max - x3min) / nz
        zslice_coord = x3min + (kz - 0.5) * dz
        push!(frames, arr[:, :, kz])
        push!(times, t)
    end

    allvals = reduce(vcat, vec.(frames))
    vmin, vmax = extrema(allvals)
    if vmax - vmin < 1e-10
        vmin, vmax = vmin - 1e-6, vmax + 1e-6
    end

    is3d = nx3 > 1

    function title_str(t)
        line1 = "Mesh $(nx1)x$(nx2)x$(nx3) zones  |  MeshBlock $(mb1)x$(mb2)x$(mb3) " *
                "($(nblocks[1])x$(nblocks[2])x$(nblocks[3]) blocks)"
        line2 = if is3d
            "Domain [$x1min,$x1max] x [$x2min,$x2max] x [$x3min,$x3max], " *
            "z-slice ≈ $(round(zslice_coord, digits=3))"
        else
            "Domain [$x1min,$x1max] x [$x2min,$x2max] x [$x3min,$x3max]"
        end
        line3 = "Ωt = $(round(omega0 * t, digits=2))"
        return line1 * "\n" * line2 * "\n" * line3
    end

    fig = Figure(size=(650, 700))
    title_obs = Observable(title_str(times[1]))
    ax = Axis(fig[1, 1], xlabel="x1", ylabel="x2", aspect=DataAspect(), title=title_obs)
    x1edges = range(x1min, x1max, length=nx1 + 1)
    x2edges = range(x2min, x2max, length=nx2 + 1)
    frame_obs = Observable(frames[1])
    hm = heatmap!(ax, x1edges, x2edges, frame_obs, colormap=:RdBu, colorrange=(vmin, vmax))
    Colorbar(fig[1, 2], hm, label=field)

    # overlay root-level MeshBlock boundaries (thin gray lines)
    blockw1 = (x1max - x1min) / nblocks[1]
    blockw2 = (x2max - x2min) / nblocks[2]
    vlines!(ax, [x1min + i * blockw1 for i in 1:nblocks[1]-1],
            color=(:gray30, 0.6), linewidth=0.8)
    hlines!(ax, [x2min + j * blockw2 for j in 1:nblocks[2]-1],
            color=(:gray30, 0.6), linewidth=0.8)

    # overlay refined-region outlines (from <refined_regionN> blocks, if any)
    for (secname, sec) in params
        startswith(secname, "refined_region") || continue
        r1min, r1max = parse(Float64, sec["x1min"]), parse(Float64, sec["x1max"])
        r2min, r2max = parse(Float64, sec["x2min"]), parse(Float64, sec["x2max"])
        lines!(ax, [r1min, r1max, r1max, r1min, r1min],
                [r2min, r2min, r2max, r2max, r2min],
                color=:limegreen, linewidth=2.5)
        text!(ax, r1min, r2max, text="level $(sec["level"])",
              color=:limegreen, fontsize=11, align=(:left, :top), offset=(3, -2))
    end

    out_path = joinpath(run_dir, "$(field)_movie.mp4")
    record(fig, out_path, eachindex(frames); framerate=15) do i
        frame_obs[] = frames[i]
        title_obs[] = title_str(times[i])
    end
    println("Wrote $out_path ($(length(frames)) frames)")
end

main()
