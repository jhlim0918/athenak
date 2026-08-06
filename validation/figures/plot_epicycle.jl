# Radial velocity of the epicycle test over 20 orbits: uniform grid vs annular
# (full-x2) level-1 ring with FARGO. The ipert=1 initial condition is spatially
# uniform, vx(0) = amp, vy'(0) = 0, so the exact solution of
#     dvx/dt = 2*Omega*vy',   dvy'/dt = -(2-q)*Omega*vx
# is  vx(t) = amp*cos(kappa t),  vy'(t) = -(amp*kappa/2Omega)*sin(kappa t),
# with kappa^2 = 2(2-q)Omega^2 = 1 for q=3/2, Omega=1 (so kappa = Omega).
using CairoMakie, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13, Axis=(xgridcolor=(:gray,0.25), ygridcolor=(:gray,0.25))))
const C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

RUN = joinpath(@__DIR__, "..", "run_epicycle")
OUT = @__DIR__

function read_hst(f)
    rows = Float64[]; ncol = 0
    for ln in eachline(f)
        startswith(strip(ln), "#") && continue
        v = parse.(Float64, split(ln)); ncol = length(v); append!(rows, v)
    end
    permutedims(reshape(rows, ncol, :))
end

u = read_hst(joinpath(RUN, "epi_unif.hydro.hst"))
r = read_hst(joinpath(RUN, "epi_ring.hydro.hst"))

amp, Ω, q = 0.1, 1.0, 1.5
κ = sqrt(2*(2-q))*Ω          # = 1
cs = sqrt((5/3)*1.0/1.0)     # ideal EOS, p0 = d0 = 1
Torb = 2π/Ω

# volume-averaged velocities = (momentum)/mass  (the ipert=1 field is spatially uniform)
vx_u = u[:,4] ./ u[:,3];  vy_u = u[:,5] ./ u[:,3];  t_u = u[:,1]
vx_r = r[:,4] ./ r[:,3];  vy_r = r[:,5] ./ r[:,3];  t_r = r[:,1]
ana(t) = amp .* cos.(κ .* t)
# epicyclic invariant: vx = A cos(κt+φ), vy' = -(Aκ/2Ω) sin(κt+φ)  =>  A² = vx² + (2Ω/κ)²vy'²
# It isolates numerical damping/growth (drift in A) from phase error.
envel(vx, vy) = sqrt.(vx.^2 .+ (2Ω/κ)^2 .* vy.^2)
A_u, A_r = envel(vx_u, vy_u), envel(vx_r, vy_r)

@printf("cs = %.4f (ideal EOS, p0=d0=1);  amp = %.3f code units = %.4f cs\n", cs, amp, amp/cs)
@printf("orbits covered: %.2f   (kappa = %.3f = Omega, so epicyclic period = T_orb)\n",
        t_u[end]/Torb, κ)
for (tag, t, v, A) in (("uniform", t_u, vx_u, A_u), ("ring+FARGO", t_r, vx_r, A_r))
    @printf("%-11s  vx range [%+.6f, %+.6f]   amplitude drift over 20 orbits %+.3e (%.3f%%)   max|vx-analytic| = %.2e\n",
            tag, minimum(v), maximum(v), A[end]-A[1], 100*(A[end]-A[1])/amp,
            maximum(abs.(v .- ana(t))))
end
n = min(length(t_u), length(t_r))
@printf("ring vs uniform: max|Δvx| = %.3e (%.2f%% of amp)\n",
        maximum(abs.(vx_u[1:n] .- vx_r[1:n])),
        100*maximum(abs.(vx_u[1:n] .- vx_r[1:n]))/amp)

fig = Figure(size=(880, 620))

# (a) full 20-orbit trace, normalized to the initial amplitude
ax1 = Axis(fig[1,1]; xlabel="t / T_orb", ylabel="vₓ / amp",
           title="Epicyclic radial velocity over 20 orbits (amp = $(amp), κ = Ω)")
hlines!(ax1, [-1, 1]; color=(:gray,0.55), linestyle=:dash)
lines!(ax1, t_u ./ Torb, vx_u ./ amp; color=C1, linewidth=1.2, label="uniform grid")
lines!(ax1, t_r ./ Torb, vx_r ./ amp; color=C3, linewidth=1.2, linestyle=:dash,
       label="annular ring + FARGO")
axislegend(ax1; position=:rt, framevisible=false, orientation=:horizontal)
ylims!(ax1, -1.35, 1.35)

# (b) zoom on the last two orbits, against the analytic solution
sel_u = t_u ./ Torb .>= 18;  sel_r = t_r ./ Torb .>= 18
tf = range(18*Torb, t_u[end]; length=600)
ax2 = Axis(fig[2,1]; xlabel="t / T_orb", ylabel="vₓ / amp",
           title="last two orbits vs analytic amp·cos(κt)", height=150)
lines!(ax2, tf ./ Torb, ana(tf) ./ amp; color=C2, linewidth=2.5, label="analytic")
scatter!(ax2, t_u[sel_u] ./ Torb, vx_u[sel_u] ./ amp; color=C1, markersize=6,
         label="uniform")
scatter!(ax2, t_r[sel_r] ./ Torb, vx_r[sel_r] ./ amp; color=C3, marker=:rect,
         markersize=5, label="ring")
axislegend(ax2; position=:rt, framevisible=false, orientation=:horizontal)

# (c) epicyclic amplitude invariant: flat => no numerical damping or growth
ax3 = Axis(fig[3,1]; xlabel="t / T_orb", ylabel="A(t) / amp", height=150,
           title="amplitude invariant  A = √(vₓ² + (2Ω/κ)²v_y'²)  — flat means no damping")
hlines!(ax3, [1.0]; color=(:gray,0.55), linestyle=:dash)
lines!(ax3, t_u ./ Torb, A_u ./ amp; color=C1, linewidth=1.5, label="uniform")
lines!(ax3, t_r ./ Torb, A_r ./ amp; color=C3, linewidth=1.5, linestyle=:dash,
       label="ring + FARGO")
axislegend(ax3; position=:rb, framevisible=false, orientation=:horizontal)

save(joinpath(OUT, "mg_epicycle_20orbits.png"), fig)
println("wrote mg_epicycle_20orbits.png")
