#!/usr/bin/env julia
# Side-by-side mp4 movie of a field from two or more AthenaK runs with .bin output
# (z≈0 slice). Correct for SMR/AMR runs, unlike the legacy-VTK movie scripts.
# MeshBlock outlines from the file itself: gray = root, green = level 1.
#
# Usage: julia scripts/make_bin_movie_compare.jl <run1> <run2> [<run3> ...] [field]
#                                                [--grid]
#   <run> is either  <dir>              (its bin/ holds one output family), or
#                    <dir>:<basename>   (pick one family from a shared bin/ dir,
#                                        e.g. validation/run:shwave2_nofargo_smr)
#   --grid lays the panels out in two rows (row-major) instead of one row --
#   e.g. four runs become a 2x2 grid. Output name gains a _grid suffix.
# All runs must share the output cadence. Writes
# <field>_compare_<label1>_vs_<label2>...mp4 into the first run's parent directory.

using CairoMakie
using Statistics: quantile, median, mean
include(joinpath(@__DIR__, "athenak_bin.jl"))

struct RunFrames
    label::String
    fd0::BinFileData                 # first dump (for domain geometry)
    frames::Vector{Matrix{Float64}}
    times::Vector{Float64}
    # per-frame MeshBlock outlines (AMR changes the mesh between dumps)
    outlines::Vector{Vector{Tuple{NTuple{4,Float64},Int}}}
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
    outlines = Vector{Tuple{NTuple{4,Float64},Int}}[]
    fd0 = read_bin(files[1])
    for f in files
        fd = read_bin(f)
        img, _, _ = assemble_root_slice(fd, field)
        push!(frames, img)
        push!(times, fd.time)
        push!(outlines, block_outlines(fd))
    end
    label = basename_sel === nothing ? basename(rstrip(run_dir, '/')) : basename_sel
    return RunFrames(label, fd0, frames, times, outlines)
end

"Per-frame part of a panel title: run label + MeshBlock layout of frame i."
function panel_header(run::RunFrames, i)
    fd = run.fd0
    levs_i = [lev for (_, lev) in run.outlines[i]]
    levs = sort(unique(levs_i))
    per = join(["$(count(==(l), levs_i)) lev-$l" for l in levs], " + ")
    "$(run.label)\nMeshBlock $(fd.nx_mb[1])x$(fd.nx_mb[2])x$(fd.nx_mb[3]) blocks: $per"
end

const LEV_COLORS = ((:gray30, 0.6), (:limegreen, 0.9), (:darkorange, 0.9), (:red, 0.9))
const MAXLEV_DRAWN = length(LEV_COLORS) - 1

"NaN-separated outline vertices of frame i for each drawn level."
function outline_segments(run::RunFrames, i)
    segs = [Point2f[] for _ in 0:MAXLEV_DRAWN]
    for (rect, lev) in run.outlines[i]
        r1min, r1max, r2min, r2max = rect
        v = segs[min(lev, MAXLEV_DRAWN) + 1]
        push!(v, Point2f(r1min, r2min), Point2f(r1max, r2min), Point2f(r1max, r2max),
                 Point2f(r1min, r2max), Point2f(r1min, r2min), Point2f(NaN, NaN))
    end
    return segs
end

function setup_panel!(fig, pos, run::RunFrames, frame_obs, seg_obs, vmin, vmax,
                      title_obs)
    fd = run.fd0
    ax = Axis(fig[pos[1], pos[2]], xlabel="x1", ylabel="x2", aspect=DataAspect(),
              title=title_obs, titlesize=12)
    x1edges = range(fd.x1min, fd.x1max, length=fd.Nx1+1)
    x2edges = range(fd.x2min, fd.x2max, length=fd.Nx2+1)
    hm = heatmap!(ax, x1edges, x2edges, frame_obs,
                   colormap=:RdBu, colorrange=(vmin, vmax))
    for l in 0:MAXLEV_DRAWN
        lines!(ax, seg_obs[l+1], color=LEV_COLORS[l+1],
               linewidth=(l == 0 ? 0.8 : 1.2))
    end
    return hm
end

function main()
    # trailing argument is the field name if it is not an existing run directory
    args = copy(ARGS)
    grid = "--grid" in args
    args = filter(!=("--grid"), args)
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
    # floor the half-width relative to the field magnitude: a field that is uniform to
    # roundoff (e.g. epicycle density) otherwise degenerates to cmin == cmax in Float32
    halfw = max(quantile(abs.(reduce(vcat, vec.(runs[1].frames[1:nframes])) .- mid),
                         0.995), 1e-6*abs(mid), 1e-10)
    vmin, vmax = mid - halfw, mid + halfw

    # panel layout: one row, or (--grid) two rows filled row-major
    ncols = grid ? cld(n, 2) : n
    nrows = grid ? 2 : 1
    positions = [(div(i-1, ncols) + 1, mod1(i, ncols)) for i in 1:n]
    fig = Figure(size=(360 + 390*ncols, 60 + 580*nrows))
    obs = [Observable(r.frames[1]) for r in runs]
    segs = [[Observable(s) for s in outline_segments(r, 1)] for r in runs]
    # per-frame data range in each title: a panel that saturates the shared color
    # scale (e.g. a run corrupted by an unsupported configuration) still reports
    # how far out of range it actually is
    titles = [Observable(panel_header(r, 1)) for r in runs]
    hm = nothing
    for (i, r) in enumerate(runs)
        h = setup_panel!(fig, positions[i], r, obs[i], segs[i], vmin, vmax, titles[i])
        i == 1 && (hm = h)
    end
    Colorbar(fig[1:nrows, ncols+1], hm, label=field)
    suptitle = Observable("")
    Label(fig[0, 1:ncols+1], suptitle, fontsize=16, font=:bold)

    tlabel(t) = omega0 === nothing ? "t = $(round(t, digits=2))" :
        "Ωt = $(round(omega0 * t, digits=2))"

    out_path = joinpath(dirname(first(parse_run_spec(args[1]))),
                        "$(field)_compare_" * join([r.label for r in runs], "_vs_") *
                        (grid ? "_grid" : "") * ".mp4")
    record(fig, out_path, 1:nframes; framerate=15) do i
        for (k, r) in enumerate(runs)
            obs[k][] = r.frames[i]
            for (l, s) in enumerate(outline_segments(r, i))
                segs[k][l][] = s
            end
            lo, hi = extrema(r.frames[i])
            oos = (lo < vmin - 1e-30) || (hi > vmax + 1e-30)   # off the shared scale
            titles[k][] = panel_header(r, i) *
                "\nrange: [" * string(round(lo, sigdigits=5)) * ", " *
                string(round(hi, sigdigits=5)) * "]" * (oos ? "  ** OFF SCALE **" : "")
        end
        suptitle[] = tlabel(runs[1].times[i])
    end
    println("Wrote $out_path ($nframes frames, $n panels)")
end

main()
