# Why the standard-resolution beta = 80 run failed.
#
# Six panels, beta = 10 and 40 shown as healthy controls throughout:
#   (a) Toomre Q                    (b) <cs>_rho          (c) M/M0 and E/E0
#   (d) <rho>_xy(z) vs time         (e) dM/dt and where the mass comes from
#   (f) halo density at |z| = 6H
#
# Q, <cs>_rho and dM/dt come from the user history (0.25/Omega); the vertical
# profiles and the outer-layer flux come from scripts/extract_gt_stress.py's
# gt_stress.npz (10/Omega).
#
# Panel (e) is the airtight part of the argument.  The x1/x2 boundaries are
# (shear-)periodic and conserve mass identically, and from Omega t = 100 onward
# NO cell in the beta = 80 box sits on the density floor (min rho = 4.4e-3 at
# t = 100 rising to 0.53 at t = 400, checked directly in the dumps), so the floor
# creates nothing either: every gram of the 8x mass gain has to cross the z faces.
# The dashed proxy -(<rho vz>_top - <rho vz>_bot) * Lx * Ly is the vertical flux in
# the outermost CELL layer, not the face flux the diode BC actually applies -- it
# runs about 2x the true dM/dt -- but it fixes the sign: inward.
#
# Usage: julia validation/figures/plot_sc14_b80_diagnosis.jl
using CairoMakie, Printf, DelimitedFiles, Statistics, PyCall
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))

const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run")
const G    = 0.3384
const LXY  = 64.0 * 64.0

runs = [(10.0, "gt_sc14_stndrd_b10", RGBf(0.17, 0.42, 0.72)),
        (40.0, "gt_sc14_stndrd_b40", RGBf(0.13, 0.55, 0.35)),
        (80.0, "gt_sc14_stndrd_b80", RGBf(0.80, 0.14, 0.22))]

function hist(dir)
    f = only(filter(x -> endswith(x, ".user.hst"), readdir(joinpath(RUN, dir), join=true)))
    d = readdlm(f, comments=true, comment_char='#')
    keep = falses(size(d, 1)); tmin = Inf
    for i in size(d, 1):-1:1
        d[i, 1] < tmin && (keep[i] = true; tmin = d[i, 1])
    end
    d = d[keep, :]
    (t=d[:, 1], mass=d[:, 3], cs=d[:, 4] ./ d[:, 3],
     Q=(d[:, 4] ./ d[:, 3]) ./ (pi * G * d[:, 3] / LXY))
end

function stress(dir)
    d = pyimport("numpy").load(joinpath(RUN, dir, "gt_stress.npz"))
    g(k) = convert(Vector{Float64}, get(d, k))
    m(k) = convert(Matrix{Float64}, get(d, k))
    (t=g("time"), z=g("x3"), rho_z=m("rho_z"), mflux=m("mflux_z"), mass=g("mass"))
end

function etot(dir)
    f = joinpath(RUN, dir, "GravitoTurb.hydro.hst")
    d = readdlm(f, comments=true, comment_char='#')
    keep = falses(size(d, 1)); tmin = Inf
    for i in size(d, 1):-1:1
        d[i, 1] < tmin && (keep[i] = true; tmin = d[i, 1])
    end
    d = d[keep, :]
    (t=d[:, 1], E=d[:, 7])
end

H = Dict(d => hist(d) for (_, d, _) in runs)
S = Dict(d => stress(d) for (_, d, _) in runs)
E = Dict(d => etot(d)   for (_, d, _) in runs)

fig = Figure(size=(1500, 940))
Label(fig[0, 1:3], "What went wrong at β = 80 (standard resolution): a z-boundary mass runaway",
      fontsize=20, font=:bold, padding=(0, 0, 6, 0))

ax(gp; kw...) = Axis(gp; xlabel="Ω t", titlealign=:left, titlefont=:bold,
                     titlesize=15, xgridcolor=(:gray, 0.15),
                     ygridcolor=(:gray, 0.15), kw...)

# ---- (a) Toomre Q
a1 = ax(fig[1, 1]; ylabel="Toomre Q", title="(a)  Q self-regulates until it doesn't")
for (b, d, c) in runs
    lines!(a1, H[d].t, H[d].Q, color=(c, 0.9), linewidth=1.8, label="β = $(Int(b))")
end
hlines!(a1, [1.33], color=(:gray, 0.7), linestyle=:dot)
text!(a1, 200, 0.40; text="dotted: SC14 β=10 plateau, Q ≈ 1.33", color=:gray,
      fontsize=11, align=(:center, :bottom))
ylims!(a1, 0.3, 3.2); axislegend(a1, position=:rt, framevisible=false, labelsize=12)

# ---- (b) sound speed
a2 = ax(fig[1, 2]; ylabel="⟨c_s⟩_ρ", yscale=log10,
        yticks=([2, 3, 5, 10, 20], ["2", "3", "5", "10", "20"]),
        title="(b)  β = 80 never stops heating")
for (b, d, c) in runs
    lines!(a2, H[d].t, H[d].cs, color=(c, 0.9), linewidth=1.8)
end
hlines!(a2, [2.26], color=(:gray, 0.7), linestyle=:dot)
text!(a2, 200, 1.58; text="dotted: SC14 tc=80, ⟨c_s⟩ = 2.26", color=:gray, fontsize=11,
      align=(:center, :bottom))
ylims!(a2, 1.5, 25)

# ---- (c) mass and energy
a3 = ax(fig[1, 3]; ylabel="M / M₀   and   E / E₀", yscale=log10,
        title="(c)  mass ×9, energy ×870")
for (b, d, c) in runs
    lines!(a3, H[d].t, H[d].mass ./ H[d].mass[1], color=(c, 0.9), linewidth=1.8)
    lines!(a3, E[d].t, E[d].E ./ E[d].E[1], color=(c, 0.55), linewidth=1.6,
           linestyle=:dash)
end
hlines!(a3, [1.0], color=(:gray, 0.6), linestyle=:dot)
text!(a3, 390, 6.0; text="M/M₀", color=runs[3][3], fontsize=12.5, align=(:right, :top))
text!(a3, 300, 2.2e3; text="E/E₀ (dashed)", color=runs[3][3], fontsize=12.5,
      align=(:center, :top))
ylims!(a3, 0.5, 3e3)

# ---- (d) vertical profiles of beta = 80 over time
a4 = ax(fig[2, 1]; xlabel="z / H", ylabel="⟨ρ⟩_xy", yscale=log10,
        title="(d)  β = 80: the box fills from the z faces inward")
s80 = S["gt_sc14_stndrd_b80"]
tsel = [0.0, 100.0, 200.0, 300.0, 350.0, 400.0]
cmap = cgrad(:viridis, length(tsel); categorical=true, rev=true)
for (n, tt) in enumerate(tsel)
    i = argmin(abs.(s80.t .- tt))
    lines!(a4, s80.z, s80.rho_z[i, :], color=cmap[n], linewidth=2,
           label=@sprintf("Ωt = %.0f", s80.t[i]))
end
s10 = S["gt_sc14_stndrd_b10"]
i10 = argmin(abs.(s10.t .- 300.0))
lines!(a4, s10.z, s10.rho_z[i10, :], color=(:black, 0.75), linewidth=2,
       linestyle=:dash, label="β = 10, Ωt = 300")
hlines!(a4, [1e-4], color=(:gray, 0.7), linestyle=:dot)
text!(a4, 0, 8.0e-5; text="density floor 10⁻⁴ ρ₀", color=:gray, fontsize=10.5,
      align=(:center, :top))
ylims!(a4, 2.5e-5, 60.0)
axislegend(a4, position=:lt, framevisible=false, labelsize=10.5, rowgap=-2, nbanks=2)

# ---- (e) dM/dt: exact, plus the outer-layer flux proxy
smoothdt(t, y; w=5.0) = [((m = abs.(t .- t[i]) .<= w);
                          (sum(m) < 3 ? 0.0 :
                           ((tt = t[m]; yy = y[m]);
                            sum((tt .- mean(tt)) .* (yy .- mean(yy))) /
                            sum((tt .- mean(tt)).^2)))) for i in eachindex(t)]
a5 = ax(fig[2, 2]; ylabel="dM/dt",
        title="(e)  mass is entering through the z faces")
for (b, d, c) in runs
    lines!(a5, H[d].t, smoothdt(H[d].t, H[d].mass), color=(c, 0.95), linewidth=1.9,
           label="β = $(Int(b)),  exact dM/dt")
    s = S[d]
    flux = -(s.mflux[:, end] .- s.mflux[:, 1]) .* LXY
    lines!(a5, s.t, flux, color=(c, 0.45), linewidth=1.5, linestyle=:dash)
end
hlines!(a5, [0.0], color=:black, linewidth=1)
ylims!(a5, -60, 320)
text!(a5, 398, -55; text="> 0  ⇒  the box is gaining mass\ndashed: outermost-cell-layer flux (≈2× dM/dt)",
      color=:gray30, fontsize=11, align=(:right, :bottom))
axislegend(a5, position=:lt, framevisible=false, labelsize=11)

# ---- (f) halo density at the z face
a6 = ax(fig[2, 3]; ylabel="⟨ρ⟩_xy at |z| = 6H", yscale=log10,
        title="(f)  once the face leaves the floor, it runs away")
for (b, d, c) in runs
    s = S[d]
    lines!(a6, s.t, 0.5 .* (s.rho_z[:, 1] .+ s.rho_z[:, end]), color=(c, 0.9),
           linewidth=1.8)
    scatter!(a6, s.t, 0.5 .* (s.rho_z[:, 1] .+ s.rho_z[:, end]), color=c, markersize=5)
end
hlines!(a6, [1e-4], color=(:gray, 0.7), linestyle=:dot)
text!(a6, 400, 1.15e-4; text="density floor 10⁻⁴ ρ₀", color=:gray, fontsize=11,
      align=(:right, :bottom))
ylims!(a6, 5e-5, 3.0)

for a in (a1, a2, a3, a5, a6)
    xlims!(a, 0, 405)
end

Label(fig[3, 1:3],
      "β = 80 was started from the cold Q₀ = 1 initial condition instead of being morphed out of β = 40 as SC14 do.  The initial collapse leaves ⟨c_s⟩ ≈ 3.6 (b), which β = 80 cooling is far too\n" *
      "slow to radiate away; the puffed-up disk lifts the density at |z| = 6H by four orders of magnitude off the floor (f); past Ωt ≈ 100 the z boundaries become a net mass source (e), feeding a\n" *
      "~130/Ω e-folding runaway in mass and energy (c).  The x/y boundaries are periodic and no cell is on the density floor after Ωt = 100, so the z faces are the only possible source.\n" *
      "β = 40 sits on the same slope — its halo is 10× above the floor and its flux proxy is weakly positive — but it stays bounded; β = 10 keeps its halo near the floor and net-loses mass.",
      fontsize=12, color=:gray30, justification=:left, lineheight=1.3,
      padding=(0, 0, 4, 4))
rowgap!(fig.layout, 8); colgap!(fig.layout, 14)

out = joinpath(HERE, "sc14_b80_diagnosis.png")
save(out, fig; px_per_unit=2)
println("wrote $out")
