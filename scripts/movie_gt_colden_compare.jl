#!/usr/bin/env julia
# Side-by-side surface-density movie of two runs, frames matched by simulation time,
# one shared color scale, refined MeshBlocks outlined wherever a run has them.
#
#   Sigma_g(x,y) = int rho dz on the root grid (conservative restriction keeps the
#   column integral exact), log10(Sigma/Sigma0) with Sigma0 = 2 rho0 H.
#
# Usage: julia scripts/movie_gt_colden_compare.jl <runA> <runB> [out.mp4]
#   <run> is a run directory (its bin/ is used) or a bin/ directory.
# Optional env: LABELA/LABELB (panel titles), VMIN/VMAX (log10 Sigma/Sigma0; default
#   0.2/99.9 percentiles over BOTH runs), SIGMA0 (2.0), FPS (12), PX (1.5),
#   OUTLINE (draw MeshBlocks at level >= this; default 1), STATS=<csv> to also write
#   per-frame t, rho_max (raw blocks), Sigma_max, mass, nmb for both runs.
using CairoMakie, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=16))
include(joinpath(@__DIR__, "athenak_bin.jl"))

bindir(a) = (a = rstrip(a, '/'); isdir(joinpath(a, "bin")) ? joinpath(a, "bin") : a)
dA, dB = bindir(ARGS[1]), bindir(ARGS[2])
out  = length(ARGS) > 2 ? ARGS[3] : joinpath(dirname(dA), "colden_compare.mp4")
labA = get(ENV, "LABELA", basename(dirname(dA))); labB = get(ENV, "LABELB", basename(dirname(dB)))
sig0 = parse(Float64, get(ENV, "SIGMA0", "2.0")); fps = parse(Int, get(ENV, "FPS", "12"))
px   = parse(Float64, get(ENV, "PX", "1.5"));     olev = parse(Int, get(ENV, "OUTLINE", "1"))
stats = get(ENV, "STATS", "")

dumps(d) = sort(filter(f -> occursin("hydro_w", f) && endswith(f, ".bin"), readdir(d, join=true)))
# cheap header time so frames can be matched before reading any data
function htime(f)
    t = open(f) do io                      # the do-block's value IS the result:
        for _ in 1:40                      # a bare `return` inside it would be discarded
            l = readline(io)
            startswith(strip(l), "time") && return parse(Float64, split(l, "=")[2])
        end
        return nothing
    end
    t === nothing && error("no time in header of $f")
    return t
end
# Match frames by NEAREST time within a tolerance, not by equality: two runs write
# at the first cycle past each output time, and their timesteps differ (the AMR run
# has finer cells), so nominally identical dumps sit a few 1e-3 apart.
tol = parse(Float64, get(ENV, "MATCH_TOL", "0.05"))
function match_frames(fa_all, fb_all, tol)     # in a function: top-level `for` would
    ta = htime.(fa_all); tb = htime.(fb_all)   # bind the accumulators as loop-locals
    out = Tuple{Float64,String,String}[]; worst = 0.0
    for (i, t) in enumerate(ta)
        j = argmin(abs.(tb .- t)); d = abs(tb[j] - t)
        d <= tol || continue
        worst = max(worst, d)
        push!(out, (t, fa_all[i], fb_all[j]))
    end
    return out, worst, length(ta), length(tb)
end
fa_all = dumps(dA); fb_all = dumps(dB)
pairsAB, worst, na, nb = match_frames(fa_all, fb_all, tol)
isempty(pairsAB) && error("no dump times matched within $tol")
common = [p[1] for p in pairsAB]
@printf("%d matched frames of %d/%d dumps, Ωt = %.3f .. %.3f (worst |Δt| = %.1e, tol %.2f)\n",
        length(common), na, nb, common[1], common[end], worst, tol)

struct Frame; map::Matrix{Float32}; segs::Vector{Point2f}; nmb::Int; nref::Int
              smax::Float64; rmax::Float64; mass::Float64; end
function reduce_dump(f)
    fd = read_bin(f); r3 = assemble_root(fd, "dens")
    dz = (fd.x3max-fd.x3min)/fd.Nx3; dx = (fd.x1max-fd.x1min)/fd.Nx1; dy = (fd.x2max-fd.x2min)/fd.Nx2
    sig = dropdims(sum(r3, dims=3), dims=3) .* dz ./ sig0
    s = Point2f[]; k = 0
    for (r, lev) in block_outlines(fd)
        lev >= olev || continue; k += 1
        a,b,c,d = Point2f(r[1],r[3]), Point2f(r[2],r[3]), Point2f(r[2],r[4]), Point2f(r[1],r[4])
        append!(s, (a,b, b,c, c,d, d,a))
    end
    rmax = maximum(maximum(fd.mb_data["dens"][m]) for m in 1:fd.n_mbs)   # raw blocks
    xs = range(fd.x1min, fd.x1max, length=fd.Nx1); ys = range(fd.x2min, fd.x2max, length=fd.Nx2)
    Frame(Float32.(log10.(clamp.(sig, 1e-10, Inf))), s, fd.n_mbs, k, maximum(sig), rmax,
          sum(r3)*dx*dy*dz), xs, ys
end

# grid coordinates come from the first dump, outside the loop (a top-level `for`
# would otherwise bind them as loop-locals and leave them undefined afterwards)
_, xs, ys = reduce_dump(pairsAB[1][2])
FA = Frame[]; FB = Frame[]
for (n, (tt, fa_path, fb_path)) in enumerate(pairsAB)
    fa, _, _ = reduce_dump(fa_path); fb, _, _ = reduce_dump(fb_path)
    push!(FA, fa); push!(FB, fb)
    n % 20 == 0 && @printf("  %3d/%d  Ωt=%7.3f  A: nmb=%4d Σmax=%6.1f ρmax=%7.1f | B: nmb=%4d Σmax=%6.1f ρmax=%7.1f\n",
                           n, length(common), tt, fa.nmb, fa.smax, fa.rmax, fb.nmb, fb.smax, fb.rmax)
end
allv = vcat((vec(f.map) for f in vcat(FA, FB))...)
vmin = parse(Float64, get(ENV, "VMIN", string(round(quantile(allv, 0.002), digits=2))))
vmax = parse(Float64, get(ENV, "VMAX", string(round(quantile(allv, 0.999), digits=2))))
@printf("color range %.2f .. %.2f dex\n", vmin, vmax)

if !isempty(stats)
    open(stats, "w") do io
        println(io, "t,rhomax_A,sigmax_A,mass_A,nmb_A,rhomax_B,sigmax_B,mass_B,nmb_B")
        for (i, tt) in enumerate(common)
            a, b = FA[i], FB[i]
            @printf(io, "%.4f,%.6g,%.6g,%.8g,%d,%.6g,%.6g,%.8g,%d\n", tt, a.rmax, a.smax, a.mass, a.nmb,
                    b.rmax, b.smax, b.mass, b.nmb)
        end
    end
    println("wrote $stats")
end

fig = Figure(size=(1700, 900))
ttl = Observable(""); Label(fig[0, 1:3], ttl; fontsize=22, font=:bold, padding=(0,0,4,0))
subA = Observable(""); subB = Observable("")
axA = Axis(fig[1,1]; title=subA, titlesize=15, xlabel="x / H", ylabel="y / H", aspect=DataAspect(),
           xticks=-32:16:32, yticks=-32:16:32)
axB = Axis(fig[1,2]; title=subB, titlesize=15, xlabel="x / H", aspect=DataAspect(),
           xticks=-32:16:32, yticks=-32:16:32, yticklabelsvisible=false)
mA = Observable(FA[1].map); mB = Observable(FB[1].map)
sA = Observable(FA[1].segs); sB = Observable(FB[1].segs)
hm = heatmap!(axA, xs, ys, mA; colormap=:inferno, colorrange=(vmin, vmax))
heatmap!(axB, xs, ys, mB; colormap=:inferno, colorrange=(vmin, vmax))
linesegments!(axA, sA; color=(:cyan,0.85), linewidth=0.9)
linesegments!(axB, sB; color=(:cyan,0.85), linewidth=0.9)
for ax in (axA, axB); xlims!(ax, xs[1], xs[end]); ylims!(ax, ys[1], ys[end]); end
Colorbar(fig[1,3], hm; label="log₁₀ Σ_g / Σ₀", height=Relative(0.85), width=16)
Label(fig[2,1:3], "same color scale in both panels; cyan: MeshBlocks refined above the root level",
      fontsize=13, color=:gray30, padding=(0,0,2,4))
colgap!(fig.layout, 1, 8); rowgap!(fig.layout, 4)

fmt(lab, f) = f.nref > 0 ?
    @sprintf("%s     %d blocks (%d refined)     Σ_max = %.0f Σ₀,  ρ_max = %.0f ρ₀", lab, f.nmb, f.nref, f.smax, f.rmax) :
    @sprintf("%s     %d blocks     Σ_max = %.0f Σ₀,  ρ_max = %.0f ρ₀", lab, f.nmb, f.smax, f.rmax)
record(fig, out, eachindex(common); framerate=fps) do n
    mA[] = FA[n].map; mB[] = FB[n].map; sA[] = FA[n].segs; sB[] = FB[n].segs
    ttl[] = @sprintf("β = 3 from the β = 10 state:  Ωt = %.2f", common[n])
    subA[] = fmt(labA, FA[n]); subB[] = fmt(labB, FB[n])
end
println("wrote $out")
