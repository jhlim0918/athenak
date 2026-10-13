# BR_3sp health check: stage 2 of the Baehr+22 box restarted at t = 150 with THREE dust
# species (St = 0.01, 0.1, 1) at Z = 0.04 and back-reaction on.  Does each size settle to
# the scale height a single turbulent diffusivity predicts, and does the dust concentrate?
# Panels: (a) H_d/H_g per species with the settling-diffusion equilibrium marked;
# (b) the diffusivity each species implies, which should be one number;
# (c) max dust density against the Roche density; (d) the gas state (stresses, Q, mass).
# Data: validation/run/gtb_BR_3sp/ (gtb.phst, gtb.user.hst).
# Usage: julia validation/figures/plot_gt_baehr_3sp.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
const RUN = joinpath(dirname(HERE), "run", "gtb_BR_3sp")

function readhst(f)
    hdr = ""
    for l in eachline(f); if startswith(l, "#") && occursin("[1]=", l); hdr = l; break; end; end
    cols = [m.captures[1] for m in eachmatch(r"\[\d+\]=(\S+)", hdr)]
    d = readdlm(f, comments=true)
    keep = trues(size(d, 1))
    for i in 1:size(d, 1)-1
        any(d[i+1:end, 1] .<= d[i, 1] + 1e-12) && (keep[i] = false)
    end
    Dict(c => d[keep, i] for (i, c) in enumerate(cols))
end

P = readhst(joinpath(RUN, "gtb.phst"))
U = readhst(joinpath(RUN, "gtb.user.hst"))
Hg = 2.16878; fpg = 4.25231; G = fpg/(4pi); L = 55.2302
St = [0.01, 0.1, 1.0]
tp = P["time"]
Hd = [P["sig_z_$s"] ./ Hg for s in 1:3]
rhoR = 9.0/fpg                                   # Roche density, Omega = 1
# the diffusivity each species implies from H_d/H_g = sqrt(d/(d+St))
# (undefined until a species has actually settled below the gas scale height: at
# insertion H_d = H_g and the expression diverges, so mask that in)
dz = [ [h < 0.99 ? St[s]*h^2/(1-h^2) : NaN for h in Hd[s]] for s in 1:3]

tu = U["time"]; mass = U["mass"]; rcs2 = U["rho_cs2"]
aR = (2/3) .* U["wrey"] ./ rcs2; aG = (2/3) .* U["wgrv"] ./ rcs2
cs = U["rho_cs"] ./ mass; Q = cs ./ (pi*G .* mass ./ L^2)
alate = let m = tu .> tu[end]-5; sum((aR.+aG)[m])/sum(m); end

cols = [:mediumblue, :darkorange2, :firebrick]
fig = Figure(size=(1180, 760))

ax1 = Axis(fig[1,1], xlabel="t  [Ω⁻¹]", ylabel="H_d / H_g", yscale=log10,
           title="(a) settling of each size")
for s in 1:3
    lines!(ax1, tp, Hd[s], color=cols[s], linewidth=2, label=@sprintf("St = %g", St[s]))
end
# equilibrium H_d for the mean implied diffusivity
dbar = let m = tp .> tp[end]-5
    v = filter(!isnan, vcat([dz[s][m] for s in 1:3]...)); sum(v)/length(v); end
for s in 1:3
    hlines!(ax1, [sqrt(dbar/(dbar+St[s]))], color=cols[s], linestyle=:dash, linewidth=1)
end
axislegend(ax1, position=:lb, framevisible=false, labelsize=12)
text!(ax1, 0.97, 0.97, space=:relative, align=(:right,:top), fontsize=11,
      text=@sprintf("dashed: √(δ/(δ+St)),\nδ = %.4f", dbar))

ax2 = Axis(fig[1,2], xlabel="t  [Ω⁻¹]", ylabel="implied δ_z", yscale=log10,
           title="(b) one diffusivity, three sizes")
for s in 1:3
    lines!(ax2, tp, dz[s], color=cols[s], linewidth=2, label=@sprintf("St = %g", St[s]))
end
hlines!(ax2, [alate], color=:black, linestyle=:dot, linewidth=2)
text!(ax2, 0.03, 0.05, space=:relative, align=(:left,:bottom), fontsize=11,
      text=@sprintf("dotted: α_R+α_G = %.4f\nSc = α/δ ≈ %.1f", alate, alate/dbar))
axislegend(ax2, position=:rt, framevisible=false, labelsize=12)
ylims!(ax2, 1e-3, 1e-1)

ax3 = Axis(fig[2,1], xlabel="t  [Ω⁻¹]", ylabel="ρ_d,max / ρ_g,0", yscale=log10,
           title="(c) dust concentration")
lines!(ax3, tp, P["dpm_max"], color=:black, linewidth=2)
hlines!(ax3, [rhoR], color=:crimson, linestyle=:dash, linewidth=2)
text!(ax3, 0.97, 0.06, space=:relative, align=(:right,:bottom), fontsize=11, color=:crimson,
      text=@sprintf("Roche 9Ω²/4πG = %.2f", rhoR))

ax4 = Axis(fig[2,2], xlabel="t  [Ω⁻¹]", ylabel="α,  Q,  M/M₀", title="(d) the gas state")
lines!(ax4, tu, aR, color=:seagreen, linewidth=2, label="α_R")
lines!(ax4, tu, aG, color=:purple, linewidth=2, label="α_G")
lines!(ax4, tu, Q, color=:black, linewidth=2, label="Q")
lines!(ax4, tu, mass ./ mass[1], color=:gray50, linewidth=2, linestyle=:dash, label="M_gas/M₀")
axislegend(ax4, position=:rc, framevisible=false, labelsize=12)
ylims!(ax4, 0, 2.0)

Label(fig[0,:], @sprintf("BR_3sp: three species, Z = 0.04, back-reaction on, t = %.1f–%.1f",
      tp[1], tp[end]), fontsize=17, font=:bold)
out = joinpath(HERE, "gt_baehr_3sp.png")
save(out, fig)
println("wrote ", out)
for s in 1:3
    m = tp .> tp[end]-5
    v = filter(!isnan, dz[s][m])
    @printf("St=%-5g  H_d/H_g = %.4f   delta_z = %.5f\n", St[s],
            sum(Hd[s][m])/sum(m), sum(v)/length(v))
end
