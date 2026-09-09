# Phase 5 (particles + static mesh refinement).  Left: the deposited dust density across
# the x1 level boundary of the SMR NSH lattice at t = 0 (dust_dpm), with the
# level-dependent kernels of the first (restrict/inject) scheme -- the predicted 9/8 in
# the second fine cell (and 15/16 in the adjacent coarse cell) -- and with the
# finest-level deposits adopted (exactly 1).  Middle: the tide + drag settling test:
# <z>(t) of the particles that settle through the refined bar ("in") and of those that
# stay on the root mesh ("out"), uniform mesh and SMR twin, against the analytic damped
# oscillator for the three stopping times.  Right: the linA streaming-instability mode
# amplitude on the uniform 64^2 mesh and with the two middle radial columns refined, and the YJ07 growth rate.
#
# Data: validation/run/dust5/ (README there)
# Usage: julia validation/figures/plot_dust5_smr.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run", "dust5")
hst(path) = readdlm(path, comments=true, comment_char='#')
function labels(path)
    for l in eachline(path)
        occursin("[1]=", l) || continue
        return [split(t, "=")[2] for t in split(replace(l, "#" => "")) if occursin("=", t)]
    end
    String[]
end

fig = Figure(size=(1500, 460))

# --- left: interface profile of the deposited density
ax1 = Axis(fig[1, 1], xlabel="x  (level boundary at 0.25; fine cells x < 0.25)",
           ylabel="deposited dust density / uniform", title="TSC lattice deposit across the level boundary")
prof = readdlm(joinpath(RUN, "dpm", "interface_profile_old.txt"), comments=true, comment_char='#')
scatterlines!(ax1, prof[:, 1], prof[:, 2], color=:firebrick, markersize=11,
              label="level-dependent kernels (restrict/inject)")
hlines!(ax1, [1.0], color=:black, linewidth=2, label="finest-level kernel everywhere: 1.000000000000000")
hlines!(ax1, [1.125, 0.9375], color=:gray, linestyle=:dash, label="predicted 9/8 and 15/16")
axislegend(ax1, position=:lt, framevisible=false)

# --- middle: settling
ax2 = Axis(fig[1, 2], xlabel="t  (Ω⁻¹)", ylabel="⟨z⟩  (H)", title="tide + drag: settling through a refined bar")
cols = (:steelblue, :darkorange, :seagreen)
for (f, ls, tag) in ((joinpath(RUN, "settle", "settle.user.hst"), :solid, "uniform"),
                     (joinpath(RUN, "settle", "settle_smr.user.hst"), :dash, "SMR"))
    d = hst(f); lab = labels(f); col = Dict(l => i for (i, l) in enumerate(lab))
    t = d[:, col["time"]]
    for s in 1:3
        zin = d[:, col["zin_$s"]] ./ max.(d[:, col["nin_$s"]], 1)
        lines!(ax2, t, zin, color=cols[s], linestyle=ls, linewidth=2,
               label=(s == 1 ? "$tag, |x| < 1 (in)" : nothing))
        if tag == "SMR"
            zout = d[:, col["zout_$s"]] ./ max.(d[:, col["nout_$s"]], 1)
            lines!(ax2, t, zout, color=cols[s], linestyle=:dot, linewidth=2,
                   label=(s == 1 ? "SMR, |x| ≥ 1 (out, root mesh)" : nothing))
            zex = d[:, col["zex_$s"]]
            lines!(ax2, t, zex, color=:black, linewidth=1, label=(s == 1 ? "analytic" : nothing))
        end
    end
end
text!(ax2, 8.5, 1.55, text="Ωτ = 10"; color=cols[3]); text!(ax2, 8.5, 0.75, text="Ωτ = 0.1"; color=cols[1])
text!(ax2, 4.0, -0.35, text="Ωτ = 1"; color=cols[2])
axislegend(ax2, position=:rt, framevisible=false)

# --- right: linA growth
ax3 = Axis(fig[1, 3], xlabel="t  (Ω⁻¹)", ylabel="|ρ̃_p|  (mode projection)", yscale=log10,
           title="linA streaming instability, 64²: uniform vs refined columns")
for (f, lab, ls) in ((joinpath(RUN, "linA", "linA_u64.user.hst"), "uniform", :solid),
                     (joinpath(RUN, "linA", "linA_s64.user.hst"), "middle columns x ∈ [L/4, 3L/4] refined", :dash))
    isfile(f) || continue
    d = hst(f); t = d[:, 1]; amp = hypot.(d[:, 3], d[:, 4])
    m = (t .> 1.0) .& (amp .> 0)
    s = (sum(m) > 2) ? (sum((t[m] .- sum(t[m])/sum(m)) .* (log.(amp[m]) .- sum(log.(amp[m]))/sum(m))) /
                        sum((t[m] .- sum(t[m])/sum(m)).^2)) : NaN
    lines!(ax3, t, amp, linestyle=ls, linewidth=2, label=@sprintf("%s: s = %.4f", lab, s))
end
d = hst(joinpath(RUN, "linA", "linA_u64.user.hst")); t = d[:, 1]; a0 = hypot(d[1, 3], d[1, 4])
lines!(ax3, t, a0 .* exp.(0.4190204 .* t), color=:black, linestyle=:dot, label="YJ07: s = 0.4190204")
axislegend(ax3, position=:lt, framevisible=false)

save(joinpath(HERE, "dust5_smr.png"), fig, px_per_unit=1.5)
println("wrote ", joinpath(HERE, "dust5_smr.png"))
