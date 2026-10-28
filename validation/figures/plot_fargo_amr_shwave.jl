# FARGO across x2 level jumps: error maps of the JG05 vortical shwave (64^2 root, one
# level of refinement) against a uniform 128^2 run, for the full-x2 ring (old annular
# policy), a static patch covering part of x2, and a moving box (AMR). Each panel shows
# v_x(run) - v_x(128^2), z-averaged, at the finest resolution (coarse cells repeated),
# with MeshBlock outlines; the x2 level jumps of the patch are at x2 = -/+0.125.
# Run directories are given by FARGO_AMR_RUNS (default: validation/run/fargo_amr).
# Run from validation/: julia figures/plot_fargo_amr_shwave.jl
using CairoMakie
using Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const HERE = @__DIR__
const VAL = dirname(HERE)
include(joinpath(VAL, "..", "scripts", "athenak_bin.jl"))
const RUNS = get(ENV, "FARGO_AMR_RUNS", joinpath(VAL, "run", "fargo_amr"))

# z-averaged (x1,x2) map at resolution N x N; blocks at level lev < Lmax are repeated
function assemble_fine(fd::BinFileData, var::AbstractString, N::Int)
    out = fill(NaN, N, N)
    L1 = fd.x1max - fd.x1min; L2 = fd.x2max - fd.x2min
    for m in 1:fd.n_mbs
        g = fd.mb_geometry[m, :]
        a = dropdims(sum(fd.mb_data[var][m], dims=3), dims=3) ./ size(fd.mb_data[var][m], 3)
        i0 = round(Int, (g[1]-fd.x1min)/L1*N); i1 = round(Int, (g[2]-fd.x1min)/L1*N)
        j0 = round(Int, (g[3]-fd.x2min)/L2*N); j1 = round(Int, (g[4]-fd.x2min)/L2*N)
        r = (i1 - i0) ÷ size(a, 1)
        out[i0+1:i1, j0+1:j1] = repeat(a, inner=(r, r))
    end
    any(isnan, out) && error("gaps in assembled grid")
    return out
end

runs = [("ring (full x2), static", "ring", "shwave2_smr"),
        ("patch (part of x2), static", "ypatch", "shwave2_ypatch"),
        ("moving box, AMR", "amrbox", "shwave2_amrbox")]
snaps = ["00010", "00025"]
N = 128

fig = Figure(size=(1150, 760))
vmax = 0.0
maps = Dict{Tuple{Int,Int},Any}()
for (s, sn) in enumerate(snaps)
    ref = read_bin(joinpath(RUNS, "unif128", "bin", "shwave.hydro_w.$sn.bin"))
    vref = assemble_fine(ref, "velx", N)
    amp = maximum(abs, vref)
    for (c, (lab, dir, base)) in enumerate(runs)
        fd = read_bin(joinpath(RUNS, dir, "bin", "$base.hydro_w.$sn.bin"))
        e = (assemble_fine(fd, "velx", N) .- vref) ./ amp
        maps[(s, c)] = (fd, e, ref.time)
        global vmax = max(vmax, maximum(abs, e))
    end
end
x1e = range(-0.25, 0.25, length=N+1); x2e = range(-0.25, 0.25, length=N+1)
for (s, sn) in enumerate(snaps), (c, (lab, _, _)) in enumerate(runs)
    fd, e, t = maps[(s, c)]
    ax = Axis(fig[s, c], aspect=DataAspect(), xlabel="x", ylabel="y",
              title=@sprintf("%s, t = %.1f", lab, t))
    hm = heatmap!(ax, x1e, x2e, e, colormap=:balance, colorrange=(-vmax, vmax))
    for ((a1, a2, b1, b2), lev) in block_outlines(fd)
        lines!(ax, [a1, a2, a2, a1, a1], [b1, b1, b2, b2, b1],
               color=(lev > 0 ? :black : (:gray40, 0.6)), linewidth=(lev > 0 ? 1.0 : 0.6))
    end
    c == 3 && Colorbar(fig[s, 4], hm, label="(vx - vx[128²]) / max|vx[128²]|")
end
save(joinpath(HERE, "fargo_amr_shwave_err.png"), fig, px_per_unit=2)
println("wrote fargo_amr_shwave_err.png; max |err| = ", vmax)
