# Figure for the Phase 0c (AMR) section of multigrid_selfgravity_validation.pdf:
#   left  — MeshBlock count vs time for the 20-orbit moving-ring epicycle AMR run,
#           with the prescribed ring center xc(t) overlaid (rings created/destroyed
#           as xc sweeps past the vetoed boundary columns);
#   right — radial-velocity-perturbation amplitude of the 128^2 vortical shwave
#           (uniform vs static ring vs moving-ring AMR, visually indistinguishable)
#           against the linear-theory amplitude of Johnson & Gammie 2005, ApJ 635,
#           149, eq. (9): dvx = dvx0 (1+tau0^2)/(1+tau^2), tau = q*Omega*t + kx0/ky.
#           This run is the JG05 Fig. 1 (left) verification setup: tau0 = -4, so the
#           swing amplifies dvx by 17x through tau = 0 at t = 8/3.
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

# hst 1-KE -> sinusoidal-mode amplitude: KE = rho0*V*dvx^2/4  (rho0 = 1, V = 0.125,
# cs = 1, so dvx is directly in units of cs)
V0 = 0.125
dvx(h) = sqrt.(4.0 .* h[:, 7] ./ V0)
# linear theory: JG05 eq. (9), dvx = dvx0 (1+tau0^2)/(1+tau^2), tau = q*Omega*t + tau0
dvx0, q, Ω, τ0 = 1.0e-4, 1.5, 1.0, -4.0
tth = range(0, maximum(hu[:, 1]); length=400)
th(t) = dvx0 * (1 + τ0^2) / (1 + (q*Ω*t + τ0)^2)

ax2 = Axis(fig[1, 2], xlabel="Ωt", ylabel="δvₓ / cₛ",
           title="vortical shwave, 128², swing through Ωt=8/3")
lines!(ax2, collect(tth), th.(tth), color=:black, linewidth=1.4, linestyle=:dash,
       label="linear theory (JG05 eq. 9)")
lines!(ax2, hu[:, 1], dvx(hu), color=C1, linewidth=4, label="uniform")
lines!(ax2, hr[:, 1], dvx(hr), color=C3, linewidth=2.2, label="static ring (SMR)")
lines!(ax2, ha[:, 1], dvx(ha), color=C2, linewidth=1.2, linestyle=:dash,
       label="moving ring (AMR)")
axislegend(ax2, position=:lt)

for (nm, h) in (("uniform", hu), ("ring", hr), ("amr", ha))
    d = maximum(abs.(dvx(h) .- th.(h[:, 1]))) / (dvx0 * (1 + τ0^2))
    @printf("%-8s max |dvx - theory| / peak = %.3e\n", nm, d)
end

save(joinpath(HERE, "mg_amr_ring.png"), fig)
println("wrote figures/mg_amr_ring.png")
