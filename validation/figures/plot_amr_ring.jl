# Figure for the Phase 0c (AMR) section of multigrid_selfgravity_validation.pdf:
#   left  — MeshBlock count vs time for the 20-orbit moving-ring epicycle AMR run,
#           with the prescribed ring center xc(t) overlaid (rings created/destroyed
#           as xc sweeps past the vetoed boundary columns);
#   right — x-KE history of the 128^2 vortical shwave: uniform vs static ring vs
#           moving-ring AMR, visually indistinguishable (max rel. diff 2.7e-3).
# Run from validation/: julia figures/plot_amr_ring.jl
using CairoMakie
using Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13, Axis=(xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25))))
const C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

const HERE = @__DIR__
const VAL = dirname(HERE)
include(joinpath(VAL, "..", "scripts", "athenak_bin.jl"))

function read_hst(f)
    rows = Float64[]; ncol = 0
    for ln in eachline(f)
        startswith(strip(ln), "#") && continue
        v = parse.(Float64, split(ln)); ncol = length(v); append!(rows, v)
    end
    permutedims(reshape(rows, ncol, :))
end

# left panel data: block counts from the epi_amr bin family
bindir = joinpath(VAL, "run_epicycle", "bin")
files = sort(filter(f -> startswith(f, "epi_amr.hydro_w."), readdir(bindir)))
tt = Float64[]; nmb = Int[]
for f in files
    fd = read_bin(joinpath(bindir, f))
    push!(tt, fd.time); push!(nmb, fd.n_mbs)
end
orb = tt ./ (2π)

# right panel data: 128^2 shwave histories (keep last monotone segment)
seg(h) = begin
    s = 1
    for i in 2:size(h, 1); h[i, 1] <= h[i-1, 1] && (s = i); end
    h[s:end, :]
end
hu = seg(read_hst(joinpath(VAL, "run", "shw_fargo128.hydro.hst")))
hr = seg(read_hst(joinpath(VAL, "run", "ring_fargo128.hydro.hst")))
ha = seg(read_hst(joinpath(VAL, "run", "shwave2_amr128.hydro.hst")))

fig = Figure(size=(1000, 360))

ax1 = Axis(fig[1, 1], xlabel="orbits", ylabel="MeshBlocks",
           title="20-orbit epicycle, moving-ring AMR")
stairs!(ax1, orb, nmb, color=C1, step=:post, label="MeshBlocks")
ax1r = Axis(fig[1, 1], yaxisposition=:right, ylabel="ring center xc(t)",
            ylabelcolor=C2, yticklabelcolor=C2)
hidespines!(ax1r); hidexdecorations!(ax1r)
lines!(ax1r, orb, 7.0 .* sin.(tt), color=(C2, 0.55))
hlines!(ax1r, [-5, 5], color=(:gray, 0.5), linestyle=:dash)
xlims!(ax1, 0, 20); xlims!(ax1r, 0, 20)

ax2 = Axis(fig[1, 2], xlabel="t", ylabel="x-kinetic energy",
           title="vortical shwave, 128², swing through t=2.67")
lines!(ax2, hu[:, 1], hu[:, 7], color=C1, linewidth=4, label="uniform")
lines!(ax2, hr[:, 1], hr[:, 7], color=C3, linewidth=2.2, label="static ring (SMR)")
lines!(ax2, ha[:, 1], ha[:, 7], color=C2, linewidth=1.2, linestyle=:dash,
       label="moving ring (AMR)")
axislegend(ax2, position=:lt)

save(joinpath(HERE, "mg_amr_ring.png"), fig)
println("wrote figures/mg_amr_ring.png")
