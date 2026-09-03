# Phase 4a acceptance: the gas-only stratified gravito-turbulence box
# (inputs/shearing_box/gravito_turb_amr_test.athinput with refinement=none, tlim=3)
# evolved with rk2 and with imex2+ (the integrator the dust module requires), both at
# cfl 0.3.  Panels: gravitational and Reynolds stresses normalized by <rho P>
# (SC14 eq. 19), density-weighted rms velocity fluctuation, internal + total energy,
# and the relative difference imex2+ vs rk2 of the history columns.  Two different
# second-order integrators: agreement at the 1e-4 .. 5e-3 level is the expectation.
#
# Data: validation/run/dust4a/gt_twin/gt_{rk2,imex}.{hydro,user}.hst
# Usage: julia validation/figures/plot_dust4a_gt_twin.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))

const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run", "dust4a", "gt_twin")
rd(f) = readdlm(joinpath(RUN, f), comments=true, comment_char='#')
u_rk, u_im = rd("gt_rk2.user.hst"),  rd("gt_imex.user.hst")
h_rk, h_im = rd("gt_rk2.hydro.hst"), rd("gt_imex.hydro.hst")

fig = Figure(size=(1100, 800))
ax1 = Axis(fig[1, 1], xlabel="t [1/Omega]", ylabel="stress / <rho P>", title="stresses")
for (u, lab, ls) in ((u_rk, "rk2", :solid), (u_im, "imex2+", :dash))
    lines!(ax1, u[:, 1], u[:, 5] ./ u[:, 7], label="grav, "*lab, linestyle=ls, color=:firebrick)
    lines!(ax1, u[:, 1], u[:, 6] ./ u[:, 7], label="Reynolds, "*lab, linestyle=ls, color=:steelblue)
end
axislegend(ax1, position=:lt, framevisible=false)

ax2 = Axis(fig[1, 2], xlabel="t [1/Omega]", ylabel="rms dv", title="velocity fluctuation")
for (u, lab, ls) in ((u_rk, "rk2", :solid), (u_im, "imex2+", :dash))
    lines!(ax2, u[:, 1], sqrt.(u[:, 11] ./ u[:, 3]), label=lab, linestyle=ls)
end
axislegend(ax2, position=:lt, framevisible=false)

ax3 = Axis(fig[2, 1], xlabel="t [1/Omega]", ylabel="energy", title="internal (E_int) and total (E_tot)")
for (u, h, lab, ls) in ((u_rk, h_rk, "rk2", :solid), (u_im, h_im, "imex2+", :dash))
    lines!(ax3, u[:, 1], u[:, 12], label="E_int, "*lab, linestyle=ls, color=:darkorange)
    lines!(ax3, h[:, 1], h[:, 7], label="E_tot, "*lab, linestyle=ls, color=:black)
end
axislegend(ax3, position=:rt, framevisible=false)

ax4 = Axis(fig[2, 2], yscale=log10, xlabel="t [1/Omega]", ylabel="|imex2+ - rk2| / |rk2|",
           title="relative difference of history columns")
cols = [(u_rk, u_im, 3, "mass"), (u_rk, u_im, 12, "E_int"), (u_rk, u_im, 11, "rho dv^2"),
        (u_rk, u_im, 5, "rho w_grv"), (u_rk, u_im, 6, "rho w_rey")]
for (a, b, c, lab) in cols
    t = a[2:end, 1]
    rel = abs.(b[2:end, c] .- a[2:end, c]) ./ max.(abs.(a[2:end, c]), 1e-300)
    scatterlines!(ax4, t, max.(rel, 1e-9), label=lab, markersize=8)
end
axislegend(ax4, position=:rb, framevisible=false)
save(joinpath(HERE, "dust4a_gt_twin.png"), fig, px_per_unit=2)
for (a, b, c, lab) in cols
    @printf("%-10s rel diff at t=%.1f: %.2e\n", lab, a[end, 1], abs(b[end, c]-a[end, c])/abs(a[end, c]))
end
fig
