#!/usr/bin/env julia
# Side-by-side mp4 movie of a field from two or more AthenaK runs with .bin output
# (z≈0 slice). Correct for SMR/AMR runs, unlike the legacy-VTK movie scripts.
# MeshBlock outlines from the file itself: gray = root, green = level 1.
#
# Usage: julia scripts/make_bin_movie_compare.jl <run1> <run2> [<run3> ...] [field]
#   <run> is either  <dir>              (its bin/ holds one output family), or
#                    <dir>:<basename>   (pick one family from a shared bin/ dir,
#                                        e.g. validation/run:shwave2_nofargo_smr)
# All runs must share the output cadence. Writes
# <field>_compare_<label1>_vs_<label2>...mp4 into the first run's parent directory.

using CairoMakie
using Statistics: quantile, median, mean
include(joinpath(@__DIR__, "athenak_bin.jl"))

struct RunFrames
    label::String
    fd0::BinFileData                 # first dump (for geometry/outlines)
    frames::Vector{Matrix{Float64}}
    times::Vector{Float64}
end

"Split a run spec `<dir>` or `<dir>:<basename>` into (dir, basename-or-nothing)."
function parse_run_spec(spec)
    parts = split(spec, ':')
    length(parts) == 1 && return (rstrip(spec, '/'), nothing)
    length(parts) == 2 && return (rstrip(String(parts[1]), '/'), String(parts[2]))
    error("bad run spec '$spec' (expected <dir> or <dir>:<basename>)")
end

function load_run(run_dir, field; basename_sel=nothing)
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
    if basename_sel !== nothing
        # <basename> selects one output family; families are "<basename>.<vargroup>"
        good = filter(fm -> startswith(fm, basename_sel * "."), good)
        isempty(good) && error("no family '$basename_sel.*' with '$field' in " *
                               "$run_dir/bin (available: $(join(families, ", ")))")
    end
    isempty(good) && error("no .bin file family in $run_dir/bin contains '$field' " *
                           "(families: $(join(families, ", ")))")
    length(good) > 1 && error("multiple file families in $run_dir/bin contain " *
                              "'$field': $(join(good, ", ")). Disambiguate with " *
                              "<dir>:<basename>.")
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
    label = basename_sel === nothing ? basename(rstrip(run_dir, '/')) : basename_sel
    return RunFrames(label, fd0, frames, times)
end

"Static part of a panel title: run label + MeshBlock layout."
function panel_header(run::RunFrames)
    fd = run.fd0
    nb = (fd.Nx1 ÷ fd.nx_mb[1], fd.Nx2 ÷ fd.nx_mb[2])
    nfine = count(==(1), fd.mb_logical[:, 4])
    "$(run.label)\nMeshBlock $(fd.nx_mb[1])x$(fd.nx_mb[2])x$(fd.nx_mb[3]) " *
    "($(nb[1])x$(nb[2]) root" *
    (nfine > 0 ? " + $nfine level-1 blocks)" : " blocks)")
end

function setup_panel!(fig, col, run::RunFrames, frame_obs, vmin, vmax, title_obs)
    fd = run.fd0
    ax = Axis(fig[1, col], xlabel="x1", ylabel="x2", aspect=DataAspect(),
              title=title_obs, titlesize=12)
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
    # trailing argument is the field name if it is not an existing run directory
    args = copy(ARGS)
    field = "dens"
    if length(args) > 2 && !isdir(first(parse_run_spec(last(args))))
        field = pop!(args)
    end
    length(args) >= 2 || error("need at least two runs to compare")

    runs = map(args) do spec
        dir, bn = parse_run_spec(spec)
        load_run(dir, field; basename_sel=bn)
    end
    n = length(runs)
    nframes = minimum(length(r.frames) for r in runs)

    omega0 = try
        parse(Float64, _get_param(runs[1].fd0.header, "<shearing_box>", "omega0"))
    catch
        nothing
    end

    # robust shared color scale, set from the *first* run (the reference): a broken
    # run can blow up by orders of magnitude and would otherwise flatten every panel
    mid = mean(runs[1].frames[1])
    halfw = max(quantile(abs.(reduce(vcat, vec.(runs[1].frames[1:nframes])) .- mid),
                         0.995), 1e-10)
    vmin, vmax = mid - halfw, mid + halfw

    fig = Figure(size=(360 + 390*n, 640))
    obs = [Observable(r.frames[1]) for r in runs]
    # per-frame data range in each title: a panel that saturates the shared color
    # scale (e.g. a run corrupted by an unsupported configuration) still reports
    # how far out of range it actually is
    titles = [Observable(panel_header(r)) for r in runs]
    hm = nothing
    for (i, r) in enumerate(runs)
        h = setup_panel!(fig, i, r, obs[i], vmin, vmax, titles[i])
        i == 1 && (hm = h)
    end
    Colorbar(fig[1, n+1], hm, label=field)
    suptitle = Observable("")
    Label(fig[0, 1:n+1], suptitle, fontsize=16, font=:bold)

    tlabel(t) = omega0 === nothing ? "t = $(round(t, digits=2))" :
        "Ωt = $(round(omega0 * t, digits=2))"

    out_path = joinpath(dirname(first(parse_run_spec(args[1]))),
                        "$(field)_compare_" * join([r.label for r in runs], "_vs_") *
                        ".mp4")
    record(fig, out_path, 1:nframes; framerate=15) do i
        for (k, r) in enumerate(runs)
            obs[k][] = r.frames[i]
            lo, hi = extrema(r.frames[i])
            oos = (lo < vmin - 1e-30) || (hi > vmax + 1e-30)   # off the shared scale
            titles[k][] = panel_header(r) *
                "\nrange: [" * string(round(lo, sigdigits=5)) * ", " *
                string(round(hi, sigdigits=5)) * "]" * (oos ? "  ** OFF SCALE **" : "")
        end
        suptitle[] = tlabel(runs[1].times[i])
    end
    println("Wrote $out_path ($nframes frames, $n panels)")
end

main()
