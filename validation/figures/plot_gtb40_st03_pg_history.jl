# gtb40_st03_pg health check from the histories: the gas state (alpha, Q, mass), the dust
# layer (H_d, peak density against Roche), and the mean drift against NSH.
# Data: validation/run/gtb40_st03_pg/{gtb40.user.hst, gtb40.hydro.hst, gtb40.phst}.
# Usage: julia validation/figures/plot_gtb40_st03_pg_history.jl
using CairoMakie, DelimitedFiles, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
const RUN = joinpath(dirname(HERE), "run", "gtb40_st03_pg")
function readhst(f)
    hdr = ""
    for l in eachline(f); if startswith(l, "#") && occursin("[1]=", l); hdr = l; break; end; end
    cols = [m.captures[1] for m in eachmatch(r"\[\d+\]=(\S+)", hdr)]
    d = readdlm(f, comments=true)
    Dict(c => d[:, i] for (i, c) in enumerate(cols))
end
U = readhst(joinpath(RUN, "gtb40.user.hst")); H = readhst(joinpath(RUN, "gtb40.hydro.hst"))
P = readhst(joinpath(RUN, "gtb40.phst"))
Hg = 2.16878; L = 80/pi*Hg; fpg = 4.25231; G = fpg/(4pi); rhoR = 9/fpg
St = 0.3; etavk = 0.1063
t = U["time"]; rcs2 = U["rho_cs2"]
aR = (2/3) .* U["wrey"] ./ rcs2; aG = (2/3) .* U["wgrv"] ./ rcs2
Q = (U["rho_cs"] ./ U["mass"]) ./ (pi*G .* U["mass"] ./ L^2)
tp = P["time"]

fig = Figure(size=(1300, 820))
ax = Axis(fig[1,1], xlabel="t  [Ω⁻¹]", ylabel="α,  Q/10", title="(a) gas: gravito-turbulence")
lines!(ax, t, aR, color=:seagreen, label="α_R"); lines!(ax, t, aG, color=:purple, label="α_G")
lines!(ax, t, aR .+ aG, color=:black, linewidth=2, label=@sprintf("α  (mean %.4f)", mean(aR .+ aG)))
lines!(ax, t, Q ./ 10, color=:darkorange, linewidth=2, label=@sprintf("Q/10  (mean Q %.2f)", mean(Q)))
hlines!(ax, [0.0155], color=:gray, linestyle=:dash)
text!(ax, 0.98, 0.02, space=:relative, align=(:right,:bottom), fontsize=11, color=:gray40,
      text="dashed: 20/H saturated α = 0.0155")
axislegend(ax, position=:lt, framevisible=false, labelsize=11, nbanks=2)

ax = Axis(fig[1,2], xlabel="t  [Ω⁻¹]", ylabel="ρ_d,max / ρ_Roche", yscale=log10,
          title="(b) dust: peak density against Roche")
lines!(ax, tp, P["dpm_max"] ./ rhoR, color=:black, linewidth=2)
hlines!(ax, [1.0], color=:crimson, linestyle=:dash)
text!(ax, 0.98, 0.04, space=:relative, align=(:right,:bottom), fontsize=11, color=:crimson,
      text=@sprintf("Roche 9Ω²/4πG = %.2f", rhoR))

ax = Axis(fig[2,1], xlabel="t  [Ω⁻¹]", ylabel="H_d / H_g", yscale=log10,
          title="(c) dust layer thickness (St = 0.3)")
lines!(ax, tp, P["sig_z_1"] ./ Hg, color=:mediumblue, linewidth=2)
hlines!(ax, [sqrt(0.009/(0.009+St))], color=:gray, linestyle=:dash)
text!(ax, 0.98, 0.95, space=:relative, align=(:right,:top), fontsize=11, color=:gray40,
      text="dashed: √(δ/(δ+St)), δ = 0.009 from BR_3sp")

ax = Axis(fig[2,2], xlabel="t  [Ω⁻¹]", ylabel="mean velocity", title="(d) radial drift and the box-mean gas motion")
lines!(ax, tp, P["vx_avg_1"], color=:mediumblue, linewidth=2, label="dust ⟨v_x⟩")
lines!(ax, H["time"], H["1-mom"] ./ H["mass"], color=:black, linewidth=2, label="gas ⟨v_x⟩")
hlines!(ax, [-2*etavk*St/(1+St^2)], color=:mediumblue, linestyle=:dash, label="NSH v_x (ε→0)")
axislegend(ax, position=:rb, framevisible=false, labelsize=11)

Label(fig[0,:], @sprintf("gtb40_st03_pg histories, t = %.1f–%.1f Ω⁻¹  (N = %d particles, none lost)",
      tp[1], tp[end], Int(P["npart"][end])), fontsize=17, font=:bold)
save(joinpath(HERE, "gtb40_st03_pg_history.png"), fig, px_per_unit=1.5)
println("wrote gtb40_st03_pg_history.png")
