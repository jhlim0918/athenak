# Figures for fargo_amr_validation.pdf (orbital advection across x2 level jumps):
#   fargo_amr_conservation.png -- left: relative mass drift of the large-amplitude
#       compressible shwave (amp 0.1, periodic x1) on a static patch and under a moving box
#       (AMR), with and without the orbital-advection flux correction; right: MeshBlock
#       count of the vortical-shwave moving-box and moving-ring AMR runs.
#   fargo_amr_swing.png -- swing amplification with multigrid gravity: d_cos/A for the
#       uniform (128^2 and 256^2), ring and patch meshes, and the deviations from the uniform
#       256^2 run (the refined runs' finest resolution) with and without gravity.
# Run directories: FARGO_AMR_RUNS (default validation/run/fargo_amr).
# Run from validation/: julia figures/plot_fargo_amr_tests.jl
using CairoMakie
using Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13, Axis=(xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25))))
const C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

const HERE = @__DIR__
const VAL = dirname(HERE)
include(joinpath(VAL, "..", "scripts", "athenak_bin.jl"))
const RUNS = get(ENV, "FARGO_AMR_RUNS", joinpath(VAL, "run", "fargo_amr"))

function read_hst(f)
    rows = Float64[]; ncol = 0
    for ln in eachline(f)
        startswith(strip(ln), "#") && continue
        v = parse.(Float64, split(ln)); ncol = length(v); append!(rows, v)
    end
    permutedims(reshape(rows, ncol, :))
end
hst(dir, pat) = read_hst(joinpath(RUNS, dir,
                     only(filter(f -> endswith(f, pat), readdir(joinpath(RUNS, dir))))))

#---------------------------------------------------------------------------------------
# conservation
fig = Figure(size=(1100, 420))
ax = Axis(fig[1, 1], yscale=log10, xlabel="t Ω", ylabel="|M(t) − M(0)| / M(0)",
          title="mass, compressible shwave (amp 0.1), periodic x₁")
for (dir, lab, col, ls) in [("cons_fc", "static patch, flux correction", C1, :solid),
                            ("cons_nofc", "static patch, no correction", C1, :dash),
                            ("cons_amr_fc", "moving box (AMR), flux correction", C2, :solid),
                            ("cons_amr_nofc", "moving box (AMR), no correction", C2, :dash)]
    h = hst(dir, ".hydro.hst")
    d = abs.(h[:, 3] .- h[1, 3]) ./ h[1, 3]
    d[d .< 1e-17] .= 1e-17
    lines!(ax, h[2:end, 1], d[2:end], color=col, linestyle=ls, linewidth=2, label=lab)
end
ylims!(ax, 1e-17, 1e-1)
axislegend(ax, position=:rt, framevisible=false, labelsize=11, nbanks=2)

ax2 = Axis(fig[1, 2], xlabel="t Ω", ylabel="MeshBlocks",
           title="vortical shwave AMR: block count")
for (dir, base, lab, col) in [("amrring", "shwave2_amr", "moving ring (all y)", C3),
                              ("amrbox", "shwave2_amrbox", "moving box (part of y)", C4)]
    bdir = joinpath(RUNS, dir, "bin")
    files = sort(filter(f -> startswith(f, "$base.hydro_w."), readdir(bdir)))
    tt = Float64[]; nn = Int[]
    for f in files
        fd = read_bin(joinpath(bdir, f)); push!(tt, fd.time); push!(nn, fd.n_mbs)
    end
    stairs!(ax2, tt, nn, color=col, linewidth=2, label=lab, step=:post)
end
axislegend(ax2, position=:rt, framevisible=false, labelsize=11)
save(joinpath(HERE, "fargo_amr_conservation.png"), fig, px_per_unit=2)
println("wrote fargo_amr_conservation.png")

#---------------------------------------------------------------------------------------
# swing
A = 1e-4
sw(d) = hst(joinpath("swing", d), ".user.hst")
u, r, p, u2 = sw("unif"), sw("ring"), sw("patch"), sw("unif256")
un, pn, u2n = sw("unif_nog"), sw("patch_nog"), sw("unif256_nog")
fig = Figure(size=(1100, 420))
ax = Axis(fig[1, 1], xlabel="t Ω", ylabel="d_cos / A",
          title="swing amplification, multigrid gravity")
lines!(ax, u[:, 1], u[:, 3] ./ A, color=:black, linewidth=2, label="uniform 128²")
lines!(ax, u2[:, 1], u2[:, 3] ./ A, color=C3, linewidth=2, label="uniform 256²")
lines!(ax, r[:, 1], r[:, 3] ./ A, color=C1, linewidth=2, linestyle=:dash, label="ring (all y)")
lines!(ax, p[:, 1], p[:, 3] ./ A, color=C2, linewidth=2, linestyle=:dot, label="patch (part of y)")
axislegend(ax, position=:lb, framevisible=false, labelsize=11)
ax2 = Axis(fig[1, 2], xlabel="t Ω", ylabel="(d − d_256) / max|d_256|",
           title="deviation from the uniform 256² run")
nrm(x) = maximum(abs, x)
# deviations at each run's own output times, with the 256^2 reference linearly interpolated
# (the uniform 128^2 run takes twice the time step, so its outputs fall up to 0.007 apart)
function at(ref, t)
    map(t) do x
        k = clamp(searchsortedlast(ref[:, 1], x), 1, size(ref, 1) - 1)
        w = (x - ref[k, 1]) / (ref[k+1, 1] - ref[k, 1])
        (1 - w) * ref[k, 3] + w * ref[k+1, 3]
    end
end
lines!(ax2, u[:, 1], (u[:, 3] .- at(u2, u[:, 1])) ./ nrm(u2[:, 3]), color=:black, linewidth=2,
       label="uniform 128², gravity on")
lines!(ax2, r[:, 1], (r[:, 3] .- at(u2, r[:, 1])) ./ nrm(u2[:, 3]), color=C1, linewidth=2,
       label="ring, gravity on")
lines!(ax2, p[:, 1], (p[:, 3] .- at(u2, p[:, 1])) ./ nrm(u2[:, 3]), color=C2, linewidth=2,
       label="patch, gravity on")
lines!(ax2, pn[:, 1], (pn[:, 3] .- at(u2n, pn[:, 1])) ./ nrm(u2n[:, 3]), color=C2, linewidth=2,
       linestyle=:dash, label="patch, gravity off")
axislegend(ax2, position=:lb, framevisible=false, labelsize=11)
save(joinpath(HERE, "fargo_amr_swing.png"), fig, px_per_unit=2)
@printf("wrote fargo_amr_swing.png; patch peak %.6f, ring %.6f, uniform %.6f, uniform 256 %.6f\n",
        minimum(p[:, 3])/A, minimum(r[:, 3])/A, minimum(u[:, 3])/A, minimum(u2[:, 3])/A)
