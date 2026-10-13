#!/usr/bin/env julia
# Surface-density movie of an AMR run with the refined MeshBlocks outlined.
#
#   Sigma_g(x,y) = int rho dz  on the root grid (the conservative 2x restriction of
#   refined blocks leaves the column integral exact), log color scale in units of
#   Sigma_0 = 2 rho_0 H, and every MeshBlock above the root level drawn as a thin
#   rectangle -- so the movie shows where the refinement criterion is firing as the
#   disk fragments.
#
# All dumps are reduced first (a 256^2 map per frame is tiny), so the color range is
# fixed from the whole run and the movie does not breathe.
#
# Usage: julia scripts/movie_gt_colden_amr.jl <bin_dir_or_run_dir> [out.mp4]
# Optional env: VMIN/VMAX (log10 Sigma/Sigma0; default 0.2/99.9 percentiles over
#   all frames), SIGMA0 (2.0), FPS (12), PX (1.5), STRIDE (1: every dump),
#   OUTLINE (level >= this is drawn; default 1)
using CairoMakie, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=16))
include(joinpath(@__DIR__, "athenak_bin.jl"))

arg   = rstrip(ARGS[1], '/')
bdir  = isdir(joinpath(arg, "bin")) ? joinpath(arg, "bin") : arg
out   = length(ARGS) > 1 ? ARGS[2] : joinpath(dirname(bdir), "colden_amr.mp4")
sig0  = parse(Float64, get(ENV, "SIGMA0", "2.0"))
fps   = parse(Int,     get(ENV, "FPS", "12"))
px    = parse(Float64, get(ENV, "PX", "1.5"))
strd  = parse(Int,     get(ENV, "STRIDE", "1"))
olev  = parse(Int,     get(ENV, "OUTLINE", "1"))

files = sort(filter(f -> occursin("hydro_w", f) && endswith(f, ".bin"),
                    readdir(bdir, join=true)))[1:strd:end]
isempty(files) && error("no hydro_w dumps in $bdir")

# ---- pass 1: reduce every dump
times = Float64[]; maps = Matrix{Float32}[]; segs = Vector{Point2f}[]
nmb = Int[]; nref = Int[]; smax = Float64[]
local xs, ys
for (n, f) in enumerate(files)
    fd  = read_bin(f)
    rho = assemble_root(fd, "dens")
    dz  = (fd.x3max - fd.x3min)/fd.Nx3
    sig = dropdims(sum(rho, dims=3), dims=3) .* dz ./ sig0
    push!(times, fd.time); push!(maps, Float32.(log10.(clamp.(sig, 1e-10, Inf))))
    push!(nmb, fd.n_mbs); push!(smax, maximum(sig))
    # rectangles of refined blocks as line segments (4 per block, one draw call)
    s = Point2f[]; k = 0
    for (r, lev) in block_outlines(fd)
        lev >= olev || continue
        k += 1
        a, b, c, d = Point2f(r[1],r[3]), Point2f(r[2],r[3]), Point2f(r[2],r[4]), Point2f(r[1],r[4])
        append!(s, (a,b, b,c, c,d, d,a))
    end
    push!(segs, s); push!(nref, k)
    if n == 1
        global xs = range(fd.x1min, fd.x1max, length=fd.Nx1)
        global ys = range(fd.x2min, fd.x2max, length=fd.Nx2)
    end
    n % 10 == 0 && @printf("  %3d/%d  Ωt=%6.2f  nmb=%4d  refined footprints=%3d  Σmax=%6.1f\n",
                           n, length(files), fd.time, fd.n_mbs, k, maximum(sig))
end
allv = vcat((vec(m) for m in maps)...)
vmin = parse(Float64, get(ENV, "VMIN", string(round(quantile(allv, 0.002), digits=2))))
vmax = parse(Float64, get(ENV, "VMAX", string(round(quantile(allv, 0.999), digits=2))))
@printf("%d frames, Ωt = %.2f .. %.2f, color range %.2f .. %.2f dex\n",
        length(times), times[1], times[end], vmin, vmax)

# ---- pass 2: render
fig = Figure(size=(960, 900))
ttl = Observable("")
Label(fig[0, 1:2], ttl; fontsize=22, font=:bold, padding=(0,0,4,0))
ax = Axis(fig[1, 1]; xlabel="x / H", ylabel="y / H", aspect=DataAspect(),
          xticks=-32:16:32, yticks=-32:16:32)
mobs = Observable(maps[1]); sobs = Observable(segs[1])
hm = heatmap!(ax, xs, ys, mobs; colormap=:inferno, colorrange=(vmin, vmax))
linesegments!(ax, sobs; color=(:cyan, 0.85), linewidth=0.9)
xlims!(ax, xs[1], xs[end]); ylims!(ax, ys[1], ys[end])
Colorbar(fig[1, 2], hm; label="log₁₀ Σ_g / Σ₀", height=Relative(0.85), width=16)
Label(fig[2, 1:2], "cyan: MeshBlocks refined above the root level (8 cells/H)",
      fontsize=13, color=:gray30, padding=(0,0,2,4))
rowgap!(fig.layout, 4)

record(fig, out, eachindex(times); framerate=fps) do n
    mobs[] = maps[n]; sobs[] = segs[n]
    ttl[]  = @sprintf("β = 3 (AMR)   Ωt = %.2f     %d MeshBlocks, %d refined     Σ_max = %.0f Σ₀",
                      times[n], nmb[n], nref[n], smax[n])
end
println("wrote $out")
