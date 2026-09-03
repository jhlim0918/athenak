# The beta = 10 standard-resolution run in the form of Shi & Chiang (2014)
# Figure 2, left panel.
#
#   top     stresses normalized as in their eq. 19: alpha_grav = <rho w_grv>/<rho P>
#           and alpha_Reynolds = <rho w_rey>/<rho P>
#   middle  volume-averaged rms density fluctuation, delta_rho = rho - rho_bar(z)
#           (their eq. 17) relative to the local mean rho_bar(z) (their eq. 14),
#           together with the density-weighted rms velocity fluctuation
#           delta_v = sqrt(vx^2 + dvy^2 + vz^2) (their eq. 18), OFFSET BY +1 as
#           in their figure
#   bottom  Toomre Q = <cs>_rho Omega / (pi G Sigma)
#
# All curves are boxcar-smoothed over 12/Omega exactly as SC14 do; the raw signal
# is drawn faint underneath.  Stresses, velocity dispersion and Q come from the
# user history file (0.25/Omega); the density dispersion needs the full 3D field
# and so comes from the 32 .bin dumps (10/Omega), plotted with markers to make
# that cadence visible.
#
# SC14 have both a standard- and a high-resolution tc=10 run (black and red in
# their figure); only the standard-resolution run exists here, so their tabulated
# standard-resolution values are shown as dotted reference lines instead.
#
# Usage: julia validation/figures/plot_sc14_b10_fig2.jl
using CairoMakie, Printf, DelimitedFiles, Statistics, PyCall
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))

const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run")
const DIR  = "gt_sc14_stndrd_b10"
const G    = 0.3384
const LXY  = 64.0 * 64.0
const TAVG = (200.0, 300.0)      # SC14 Table 1 averaging window for tc=10

"Boxcar smoothing over `width` in Omega^-1, as in SC14 Fig. 2."
function smooth(t, y; width=12.0)
    [mean(y[abs.(t .- t[i]) .<= width/2]) for i in eachindex(t)]
end

d = readdlm(only(filter(x -> endswith(x, ".user.hst"),
                        readdir(joinpath(RUN, DIR), join=true))),
            comments=true, comment_char='#')
# restart seams rewind time and append: keep the last occurrence of each epoch
function dedup(d)
    keep = falses(size(d, 1)); tmin = Inf
    for i in size(d, 1):-1:1
        d[i, 1] < tmin && (keep[i] = true; tmin = d[i, 1])
    end
    return d[keep, :]
end
d = dedup(d)
t      = d[:, 1]
mass   = d[:, 3]
a_grv  = d[:, 5] ./ d[:, 7]
a_rey  = d[:, 6] ./ d[:, 7]
dv     = sqrt.(d[:, 11] ./ mass)                      # <dv^2>_rho^(1/2)
Q      = (d[:, 4] ./ mass) ./ (pi * G * mass / LXY)

np = pyimport("numpy")
b  = np.load(joinpath(RUN, DIR, "gt_stress.npz"))
tb    = convert(Vector{Float64}, get(b, "time"))
drho  = convert(Vector{Float64}, get(b, "drho_rms"))

win(tt, y) = mean(y[(tt .>= TAVG[1]) .& (tt .<= TAVG[2])])
@printf("beta = 10, Ωt = %.0f-%.0f averages\n", TAVG...)
@printf("  alpha_grav  %6.4f   (SC14 std-res 0.0387)\n", win(t, a_grv))
@printf("  alpha_Rey   %6.4f   (SC14 std-res 0.0188)\n", win(t, a_rey))
@printf("  <dv^2>_rho  %6.3f   (SC14 std-res 1.79)\n",   win(t, dv))
@printf("  drho/rho(z) %6.3f   (SC14: 'of order unity')\n", win(tb, drho))
@printf("  Q           %6.3f   (SC14 std-res 1.33)\n",   win(t, Q))

const CGRV = RGBf(0.80, 0.14, 0.22)
const CREY = RGBf(0.17, 0.42, 0.72)
const CRHO = RGBf(0.13, 0.55, 0.35)
const CVEL = RGBf(0.48, 0.24, 0.66)
const CQ   = RGBf(0.10, 0.15, 0.45)

fig = Figure(size=(880, 950))
Label(fig[0, 1], "β = 10 (Ω t_cool = 10), standard resolution 256×256×48 — SC14 Figure 2 form",
      fontsize=17, font=:bold, padding=(0, 0, 6, 0))

function panel(row; ylabel, xlabels=false)
    ax = Axis(fig[row, 1]; ylabel=ylabel,
              xlabel = xlabels ? "time  (Ω⁻¹)" : "",
              xticklabelsvisible = xlabels,
              xticks = 0:50:300, xgridcolor=(:gray, 0.15), ygridcolor=(:gray, 0.15))
    xlims!(ax, 0, 300)
    vspan!(ax, TAVG[1], TAVG[2]; color=(:gray, 0.10))
    return ax
end

# ---------------------------------------------------------------- stresses
ax1 = panel(1; ylabel="Stress")
for (y, c, lab) in ((a_grv, CGRV, "α gravitational"), (a_rey, CREY, "α Reynolds"))
    lines!(ax1, t, y, color=(c, 0.13), linewidth=0.8)
    lines!(ax1, t, smooth(t, y), color=c, linewidth=2.4, label=lab)
end
hlines!(ax1, [0.0387, 0.0188], color=(:gray, 0.75), linestyle=:dot)
text!(ax1, 297, 0.0395; text="SC14 0.0387", align=(:right, :bottom), color=:gray, fontsize=11)
text!(ax1, 297, 0.0196; text="SC14 0.0188", align=(:right, :bottom), color=:gray, fontsize=11)
ylims!(ax1, -0.012, 0.095)
axislegend(ax1, position=:lt, framevisible=false, labelsize=12.5)

# ---------------------------------------------------------------- dispersions
ax2 = panel(2; ylabel="Dispersion")
# the dumps are already spaced 10/Ω, so a 12/Ω boxcar would be a no-op: plot raw
lines!(ax2, tb, drho, color=CRHO, linewidth=2.2, label="⟨δρ²⟩¹ᐟ² / ρ̄(z)   (10/Ω dumps)")
scatter!(ax2, tb, drho, color=CRHO, markersize=7, strokecolor=:white, strokewidth=0.8)
lines!(ax2, t, dv .+ 1, color=(CVEL, 0.16), linewidth=0.8)
lines!(ax2, t, smooth(t, dv) .+ 1, color=CVEL, linewidth=2.4,
       label="⟨δv²⟩ρ¹ᐟ²  + 1   (offset for clarity)")
hlines!(ax2, [1.79 + 1], color=(:gray, 0.75), linestyle=:dot)
text!(ax2, 297, 2.83; text="SC14 δv = 1.79", align=(:right, :bottom), color=:gray, fontsize=11)
ylims!(ax2, 0.0, 3.4)
axislegend(ax2, position=:rb, framevisible=false, labelsize=12.5)

# ---------------------------------------------------------------- Toomre Q
ax3 = panel(3; ylabel="Q", xlabels=true)
lines!(ax3, t, Q, color=(CQ, 0.16), linewidth=0.8)
lines!(ax3, t, smooth(t, Q), color=CQ, linewidth=2.4)
hlines!(ax3, [1.33], color=(:gray, 0.75), linestyle=:dot)
text!(ax3, 297, 1.35; text="SC14 Q = 1.33", align=(:right, :bottom), color=:gray, fontsize=11)
ylims!(ax3, 0.45, 2.05)

text!(ax3, 250, 0.52; text="averaging window", align=(:center, :bottom),
      color=:gray45, fontsize=11)

linkxaxes!(ax1, ax2, ax3)
Label(fig[4, 1],
      "History-file curves (0.25/Ω) are boxcar-smoothed over 12/Ω as in SC14, with the raw signal faint underneath; the shaded band is\n" *
      "their Ωt = 200–300 averaging window.  δρ = ρ − ρ̄(z) (eq. 17), volume-averaged relative to the local mean, needs the full 3D field\n" *
      "and so comes from the 32 .bin dumps — already 10/Ω apart, so it is shown unsmoothed with markers.  Dotted lines are SC14's\n" *
      "standard-resolution tc=10 values (their Table 1).",
      fontsize=11.5, color=:gray30, justification=:left, lineheight=1.3,
      padding=(0, 0, 4, 4))
rowgap!(fig.layout, 6)

out = joinpath(HERE, "sc14_b10_fig2.png")
save(out, fig; px_per_unit=2)
println("wrote $out")
