#!/usr/bin/env julia
# Side-by-side mp4 movie of a field from two AthenaK runs with .bin output
# (z≈0 slice). Correct for SMR/AMR runs, unlike the legacy-VTK movie scripts.
# MeshBlock outlines from the file itself: gray = root, green = level 1.
#
# Usage: julia scripts/make_bin_movie_compare.jl <run_dir_left> <run_dir_right> [field]
# Reads <run_dir>/bin/*.bin; both runs must share the output cadence.
# Writes <field>_compare_<left>_vs_<right>.mp4 into the runs' parent directory.

using CairoMakie
using Statistics: quantile, median, mean
include(joinpath(@__DIR__, "athenak_bin.jl"))

struct RunFrames
    label::String
    fd0::BinFileData                 # first dump (for geometry/outlines)
    frames::Vector{Matrix{Float64}}
    times::Vector{Float64}
end

function load_run(run_dir, field)
    all_files = sort(filter(f -> endswith(f, ".bin"),
                            readdir(joinpath(run_dir, "bin"), join=true)))
    isempty(all_files) && error("no .bin files in $run_dir/bin")
    # AthenaK writes one file family per output block: <basename>.<vargroup>.#####.bin
    # (e.g. shwave2.hydro_w.00012.bin). A run directory may also hold other variable
    # groups (grav_phi, ...) or, worse, several runs. Select exactly the family whose
    # variables contain `field`, and refuse ambiguity instead of mixing runs.
    fam(f) = begin
        parts = split(basename(f), '.')
        length(parts) >= 4 ? join(parts[1:end-2], '.') : basename(f)
    end
    families = unique(fam.(all_files))
    good = filter(fm -> field in
                  read_bin(first(filter(f -> fam(f) == fm, all_files))).var_names,
                  families)
    isempty(good) && error("no .bin file family in $run_dir/bin contains '$field' " *
                           "(families: $(join(families, ", ")))")
    length(good) > 1 && error("multiple file families in $run_dir/bin contain " *
                              "'$field': $(join(good, ", ")). One run per directory.")
    files = filter(f -> fam(f) == only(good), all_files)
    frames = Matrix{Float64}[]
    times = Float64[]
    fd0 = read_bin(files[1])
    for f in files
        fd = read_bin(f)
        img, _, _ = assemble_root_slice(fd, field)
        push!(frames, img)
        push!(times, fd.time)
    end
    return RunFrames(basename(rstrip(run_dir, '/')), fd0, frames, times)
end

function setup_panel!(fig, col, run::RunFrames, frame_obs, vmin, vmax)
    fd = run.fd0
    nb = (fd.Nx1 ÷ fd.nx_mb[1], fd.Nx2 ÷ fd.nx_mb[2])
    nfine = count(==(1), fd.mb_logical[:, 4])
    subtitle = "MeshBlock $(fd.nx_mb[1])x$(fd.nx_mb[2])x$(fd.nx_mb[3]) " *
               "($(nb[1])x$(nb[2]) root" *
               (nfine > 0 ? " + $nfine level-1 blocks)" : " blocks)")
    ax = Axis(fig[1, col], xlabel="x1", ylabel="x2", aspect=DataAspect(),
              title="$(run.label)\n$subtitle", titlesize=12)
    x1edges = range(fd.x1min, fd.x1max, length=fd.Nx1+1)
    x2edges = range(fd.x2min, fd.x2max, length=fd.Nx2+1)
    hm = heatmap!(ax, x1edges, x2edges, frame_obs,
                   colormap=:RdBu, colorrange=(vmin, vmax))
    for (rect, lev) in block_outlines(fd)
        r1min, r1max, r2min, r2max = rect
        color = lev == 0 ? (:gray30, 0.6) : (:limegreen, 0.9)
        lines!(ax, [r1min, r1max, r1max, r1min, r1min],
                [r2min, r2min, r2max, r2max, r2min],
                color=color, linewidth=lev == 0 ? 0.8 : 1.2)
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

    omega0 = try
        parse(Float64, _get_param(left.fd0.header, "<shearing_box>", "omega0"))
    catch
        nothing
    end

    # robust shared color scale (median across runs of 99.5th-pct |deviation|)
    mid = mean(left.frames[1])
    devs = [quantile(abs.(reduce(vcat, vec.(r.frames[1:nframes])) .- mid), 0.995)
            for r in (left, right)]
    halfw = max(median(devs), 1e-10)
    vmin, vmax = mid - halfw, mid + halfw

    fig = Figure(size=(1150, 620))
    lframe = Observable(left.frames[1])
    rframe = Observable(right.frames[1])
    hm = setup_panel!(fig, 1, left, lframe, vmin, vmax)
    setup_panel!(fig, 2, right, rframe, vmin, vmax)
    Colorbar(fig[1, 3], hm, label=field)
    suptitle = Observable("")
    Label(fig[0, 1:3], suptitle, fontsize=16, font=:bold)

    tlabel(t) = omega0 === nothing ? "t = $(round(t, digits=2))" :
        "Ωt = $(round(omega0 * t, digits=2))"

    out_path = joinpath(dirname(left_dir),
                        "$(field)_compare_$(left.label)_vs_$(right.label).mp4")
    record(fig, out_path, 1:nframes; framerate=15) do i
        lframe[] = left.frames[i]
        rframe[] = right.frames[i]
        suptitle[] = tlabel(left.times[i])
    end
    println("Wrote $out_path ($nframes frames)")
end

main()
