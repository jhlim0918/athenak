# Stage 1 of the Baehr+22 setup (gas only, beta = 10 with the irradiation floor
# cs_irr = cs0): is the box in a saturated gravito-turbulent state?  Left: the stresses
# against the Gammie value and against the stress that balances the cooling actually
# available above the floor.  Middle: Toomre Q and the thermal state.  Right: the
# heating/cooling ratio.  Baehr's Table 1 values (noBR_L_10, averaged t = 50-80) are
# marked.  Data: validation/run/gt_baehr_stage1/ (gtb.user.hst, gtb.hydro.hst).
# Usage: julia validation/figures/plot_gt_baehr_stage1.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
const HERE = @__DIR__
const RUN = joinpath(dirname(HERE), "run", "gt_baehr_stage1")

function readhst(f)
    hdr = ""
    for l in eachline(f); if startswith(l, "#") && occursin("[1]=", l); hdr = l; break; end; end
    cols = [m.captures[1] for m in eachmatch(r"\[\d+\]=(\S+)", hdr)]
    d = readdlm(f, comments=true)
    keep = trues(size(d, 1))                       # a continued run appends: keep the last row per time
    for i in 1:size(d, 1)-1
        any(d[i+1:end, 1] .<= d[i, 1] + 1e-12) && (keep[i] = false)
    end
    Dict(c => d[keep, i] for (i, c) in enumerate(cols))
end
U = readhst(joinpath(RUN, "gtb.user.hst"))
t = U["time"]; mass = U["mass"]; cs = U["rho_cs"] ./ mass
G = 4.25231/(4pi); L = 55.2302; cs0 = 2.16878; gam = 5/3; beta = 10.0
Q = cs ./ (pi*G .* mass ./ L^2)
rcs2 = U["rho_cs2"]
aR = (2/3) .* U["wrey"] ./ rcs2; aG = (2/3) .* U["wgrv"] ./ rcs2; a = aR .+ aG
eint = U["eint"]; efl = cs0^2 .* mass ./ (gam*(gam-1)); cool = (eint .- efl) ./ beta
dE = [i == 1 ? (eint[2]-eint[1])/(t[2]-t[1]) : i == length(t) ? (eint[end]-eint[end-1])/(t[end]-t[end-1]) :
      (eint[i+1]-eint[i-1])/(t[i+1]-t[i-1]) for i in eachindex(t)]
aeq = cool ./ (2.25 .* rcs2)                      # H = (9/4) a' Omega <gamma P> = cooling
smooth(y, n) = [mean(y[max(1,i-n):min(length(y),i+n)]) for i in eachindex(y)]
using Statistics

fig = Figure(size=(1680, 470))
ax1 = Axis(fig[1, 1], xlabel="t  (Ω⁻¹)", ylabel="α", title="stresses: burst, relaxation, saturation")
band!(ax1, [110, 150], [0, 0], [0.06, 0.06], color=(:green, 0.08))
text!(ax1, 112, 0.050, text="saturated", fontsize=12, color=:gray30)
lines!(ax1, t, smooth(aG, 6), color=:crimson, linewidth=2, label="α_G (gravitational)")
lines!(ax1, t, smooth(aR, 6), color=:royalblue, linewidth=2, label="α_R (Reynolds)")
lines!(ax1, t, smooth(a, 6), color=:black, linewidth=2.5, label="α total")
lines!(ax1, t, aeq, color=:darkorange, linewidth=2, linestyle=:dash, label="α that balances the cooling")
hlines!(ax1, [0.04], color=:gray, linestyle=:dot, linewidth=2, label="Gammie, no floor")
scatter!(ax1, [130], [0.044], color=:crimson, marker=:star5, markersize=18, label="Baehr Table 1: α_G")
scatter!(ax1, [130], [0.0065], color=:royalblue, marker=:star5, markersize=18, label="Baehr Table 1: α_R")
axislegend(ax1, position=:lt, framevisible=false, labelsize=11)
ylims!(ax1, 0, 0.055)

ax2 = Axis(fig[1, 2], xlabel="t  (Ω⁻¹)", ylabel="Q,  c_s/c_s0,  e/e_floor", title="thermal state")
lines!(ax2, t, Q, color=:black, linewidth=2.5, label="Toomre Q")
lines!(ax2, t, cs ./ cs0, color=:seagreen, linewidth=2, label="c_s / c_s,irr")
lines!(ax2, t, eint ./ efl, color=:purple, linewidth=2, label="e_int / e_floor")
hlines!(ax2, [1.02], color=:gray, linestyle=:dot, linewidth=2, label="Q₀ = 1.02")
hlines!(ax2, [1.3], color=:crimson, linestyle=:dash, linewidth=2, label="Baehr Table 1: Q = 1.3")
axislegend(ax2, position=:lt, framevisible=false, labelsize=11)

ax3 = Axis(fig[1, 3], xlabel="t  (Ω⁻¹)", ylabel="heating / cooling", title="thermal budget: balanced to 0.2% for t > 110")
lines!(ax3, t, smooth((dE .+ cool) ./ max.(cool, 1e-6), 8), color=:black, linewidth=2.5)
hlines!(ax3, [1.0], color=:crimson, linestyle=:dash, linewidth=2)
vlines!(ax3, [100], color=:gray, linestyle=:dot, linewidth=2)
text!(ax3, 30, 3.2, text="net heating", fontsize=13, color=:gray30)
xlims!(ax3, 20, 150); ylims!(ax3, 0, 4)
band!(ax3, [110, 150], [0, 0], [4, 4], color=(:green, 0.08))
save(joinpath(HERE, "gt_baehr_stage1.png"), fig, px_per_unit=2)
println("wrote gt_baehr_stage1.png")
