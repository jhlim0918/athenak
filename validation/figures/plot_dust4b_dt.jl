# Phase 4b (iii): the particle transport timestep on the shear-subtracted velocity.
# Left: the timestep of the wide (Lx = 8) 3D shear-periodic NSH run under the legacy
# limit (dt_transport = full: one cell of |v_y - q*Omega*x| per step) and the new
# default (relative: peculiar velocity per cell, total velocity per MeshBlock), IMEX and
# PC2.  Middle: gas x-kinetic energy of the azimuthally structured run (mass_mod_amp =
# 0.5) over one orbit under both limits.  Right: relative differences of that history:
# new vs legacy limit (truncation level), serial vs 4 ranks (round-off).
#
# Data: validation/run/dust4b/dt/ (README in validation/run/dust4b/)
# Usage: julia validation/figures/plot_dust4b_dt.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))

const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run", "dust4b", "dt")

hst(path) = readdlm(path, comments=true, comment_char='#')

fig = Figure(size=(1500, 440))

# --- (a) timestep history of the ten-orbit wide runs
ax1 = Axis(fig[1, 1], xlabel="t  (Ω⁻¹)", ylabel="Δt", yscale=log10,
           title="wide box (Lx = 8): particle-limited step")
for (name, lab, ls) in (("wide_full", "IMEX, dt_transport = full (legacy)", :dash),
                        ("wide_rel",  "IMEX, relative (default)", :solid),
                        ("wide_pc2",  "PC2/rk2, relative", :dot))
    d = hst(joinpath(RUN, "wide", name * ".hydro.hst"))
    lines!(ax1, d[:, 1], d[:, 2], label=lab, linestyle=ls, linewidth=2.5)
    @printf("%-12s  dt = %.6e  (%d rows)\n", name, d[end, 2], size(d, 1))
end
axislegend(ax1, position=:rb, framevisible=false)

# --- (b) structured run: gas x-kinetic energy under both limits
ax2 = Axis(fig[1, 2], xlabel="t  (Ω⁻¹)", ylabel="gas x-kinetic energy",
           title="structured state (mass_mod_amp = 0.5), one orbit")
mod = Dict{String,Matrix{Float64}}()
for name in ("full", "rel", "rel_np4", "pc2", "pc2_full")
    mod[name] = hst(joinpath(RUN, "mod", "mod_" * name * ".hydro.hst"))
end
lines!(ax2, mod["full"][:, 1], mod["full"][:, 7], label="IMEX, full", linestyle=:dash, linewidth=2.5)
lines!(ax2, mod["rel"][:, 1],  mod["rel"][:, 7],  label="IMEX, relative", linewidth=2)
lines!(ax2, mod["pc2"][:, 1],  mod["pc2"][:, 7],  label="PC2, relative", linestyle=:dot, linewidth=2.5)
axislegend(ax2, position=:rt, framevisible=false)

# --- (c) relative differences of the structured history
ax3 = Axis(fig[1, 3], xlabel="t  (Ω⁻¹)", ylabel="relative difference", yscale=log10,
           title="gas x-KE: new vs legacy limit, serial vs 4 ranks")
function reldiff(a, b, col)
    n = min(size(a, 1), size(b, 1))
    t = a[1:n, 1]
    r = abs.(a[1:n, col] .- b[1:n, col]) ./ max.(abs.(a[1:n, col]), abs.(b[1:n, col]), 1e-300)
    return t[2:end], max.(r[2:end], 1e-17)
end
t, r = reldiff(mod["rel"], mod["full"], 7);         lines!(ax3, t, r, label="IMEX: relative vs full", linewidth=2)
t, r = reldiff(mod["pc2"], mod["pc2_full"], 7);     lines!(ax3, t, r, label="PC2: relative vs full", linewidth=2, linestyle=:dot)
t, r = reldiff(mod["rel_np4"], mod["rel"], 7);      lines!(ax3, t, r, label="IMEX relative: 4 ranks vs serial", linewidth=2, linestyle=:dash)
axislegend(ax3, position=:rb, framevisible=false)

save(joinpath(HERE, "dust4b_dt.png"), fig, px_per_unit=2)
println("wrote dust4b_dt.png")
fig
