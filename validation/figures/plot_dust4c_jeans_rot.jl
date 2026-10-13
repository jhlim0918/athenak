# The rotating dusty Jeans problem (unstratified shearing box, axisymmetric ky = 0 mode)
# with superparticle dust: measured growth rates against the (3N+3) eigenproblem.
# Panels: (a) the dynamical regime -- gas only (Chandrasekhar, 4piG = 2) and gas + dust
# (4piG = 1, eps = 1, ts = 1: 4piG rho0 (1+eps) = 2) against k; (b) the secular regime,
# where the gas alone is stable at every k -- back-reaction OFF at 4piG = 0.5 (the one-way
# drag mode of Ward 2000 / Youdin 2005, 2011, against the k-independent frozen-gas rate
# 4piG rho_d ts/(1 + kappa^2 ts^2)) and ON at 4piG = 0.6 (growth only because the mixture
# is rotationally unstable, 4piG rho0 (1+eps) = 1.2 > kappa^2; below that threshold, at
# 4piG = 0.5, sigma = 0 exactly); (c) the same against the stopping time at k = 0.6;
# (d) convergence and the couplings.  Data: validation/run/dust4c/jeans_rot/
# (jeans_rot_results.txt, curve_*.txt from battery.sh).
# Usage: julia validation/figures/plot_dust4c_jeans_rot.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run", "dust4c", "jeans_rot")

tab = readdlm(joinpath(RUN, "jeans_rot_results.txt"), comments=true)
name = String.(tab[:, 1]); k = Float64.(tab[:, 2]); nx1 = Int.(tab[:, 4])
sana = Float64.(tab[:, 5]); sgas = Float64.(tab[:, 7]); sdust = Float64.(tab[:, 9])
cg  = readdlm(joinpath(RUN, "curve_gas_f2.txt"), comments=true)
cd_ = readdlm(joinpath(RUN, "curve_dyn_f1_e1_ts1.txt"), comments=true)
cs  = readdlm(joinpath(RUN, "curve_sec_f0.5_e1_ts1.txt"), comments=true)
cb  = readdlm(joinpath(RUN, "curve_br_f0.6_e1_ts1.txt"), comments=true)
cts = readdlm(joinpath(RUN, "curve_sec_ts_k0.6.txt"), comments=true)
relerr(i) = abs(sgas[i]/sana[i] - 1) + 1e-16
pick(pat) = [i for i in eachindex(name) if occursin(pat, name[i])]
first_(pat) = (v = pick(pat); isempty(v) ? nothing : v[1])

fig = Figure(size=(1900, 520))
# ---- (a) dynamical
ax1 = Axis(fig[1, 1], xlabel="k  (Ω / c_s)", ylabel="σ  (Ω)", title="(a) dynamical: 4πGρ₀(1+ε₀) = 2, κ = Ω")
lines!(ax1, cg[:, 1], cg[:, 4], color=:gray30, linewidth=2, linestyle=:dot, label="gas only: s² = 4πGρ₀ − κ² − c_s²k²")
lines!(ax1, cd_[:, 1], cd_[:, 2], color=:black, linewidth=2.2, label="gas + dust (ε₀ = 1, t_s = 1), back-reaction")
lines!(ax1, cd_[:, 1], cd_[:, 3], color=:black, linewidth=2, linestyle=:dash, label="… no back-reaction")
i = pick(r"^gas_f2_k"); scatter!(ax1, k[i], sgas[i], marker=:circle, markersize=13, color=(:white, 0), strokecolor=:gray30, strokewidth=2, label="measured, gas only")
i = pick(r"^dyn_k[0-9.]+$"); scatter!(ax1, k[i], sgas[i], marker=:circle, markersize=13, color=:crimson, label="measured, gas + dust")
i = first_(r"^dyn_k0.6_nobr$"); i === nothing || scatter!(ax1, [k[i]], [sgas[i]], marker=:diamond, markersize=14, color=:darkorange, label="measured, no back-reaction")
axislegend(ax1, position=:lb, framevisible=false, labelsize=11); ylims!(ax1, 0, 1.1)
# ---- (b) secular vs k
ax2 = Axis(fig[1, 2], xlabel="k  (Ω / c_s)", ylabel="σ  (Ω)", title="(b) secular: gas stable at every k, ε₀ = 1, t_s = 1")
lines!(ax2, cs[:, 1], cs[:, 5], color=:seagreen, linewidth=2, linestyle=:dot, label="4πGρ₀ = 0.5, frozen gas: 4πGρ_d t_s/(1+κ²t_s²)")
lines!(ax2, cs[:, 1], cs[:, 3], color=:seagreen, linewidth=2.2, label="4πGρ₀ = 0.5, no back-reaction")
lines!(ax2, cb[:, 1], cb[:, 2], color=:black, linewidth=2.2, label="4πGρ₀ = 0.6, back-reaction (mixture 1.2 > κ²)")
hlines!(ax2, [0.0], color=:black, linewidth=1.5, linestyle=:dash, label="4πGρ₀ ≤ 0.5, back-reaction: σ = 0")
i = pick(r"^sec_ts1_k[0-9.]+$"); scatter!(ax2, k[i], sgas[i], marker=:circle, markersize=13, color=:seagreen, strokecolor=:black, strokewidth=1, label="measured, no back-reaction")
i = pick(r"^br_f0.6_ts1_k[0-9.]+$"); scatter!(ax2, k[i], sgas[i], marker=:circle, markersize=13, color=:crimson, label="measured, back-reaction, 4πGρ₀ = 0.6")
i = first_(r"^br_f0.55_ts1_k0.6_n128$"); i === nothing || scatter!(ax2, [k[i]], [sgas[i]], marker=:utriangle, markersize=14, color=:crimson, label="… 4πGρ₀ = 0.55 (128 cells/λ)")
i = first_(r"^br_f0.5_ts1_k0.6$"); i === nothing || scatter!(ax2, [k[i]], [sdust[i]], marker=:diamond, markersize=14, color=:crimson, label="… at the threshold (dust amplitude)")
axislegend(ax2, position=:rt, framevisible=false, labelsize=10); ylims!(ax2, -0.02, 0.46)
# ---- (c) vs ts
ax3 = Axis(fig[1, 3], xscale=log10, xlabel="κ t_s", ylabel="σ  (Ω)", title="(c) k = 0.6: the stopping-time dependence")
lines!(ax3, cts[:, 1], cts[:, 5], color=:seagreen, linewidth=2, linestyle=:dot, label="4πGρ₀ = 0.5, frozen gas")
lines!(ax3, cts[:, 1], cts[:, 3], color=:seagreen, linewidth=2.2, label="4πGρ₀ = 0.5, no back-reaction")
lines!(ax3, cts[:, 1], cts[:, 7], color=:black, linewidth=2.2, label="4πGρ₀ = 0.6, back-reaction")
for (nm, t, col) in (("sec_ts0.1_k0.6", 0.1, :seagreen), ("sec_ts1_k0.6", 1.0, :seagreen), ("sec_ts10_k0.6", 10.0, :seagreen),
                     ("br_f0.6_ts0.1_k0.6", 0.1, :crimson), ("br_f0.6_ts1_k0.6", 1.0, :crimson), ("br_f0.6_ts10_k0.6", 10.0, :crimson))
    i = first_(Regex("^" * nm * raw"$")); i === nothing && continue
    scatter!(ax3, [t], [sgas[i]], marker=:circle, markersize=13, color=col, strokecolor=:black, strokewidth=1)
end
axislegend(ax3, position=:lt, framevisible=false, labelsize=10)
# ---- (d) convergence
ax4 = Axis(fig[1, 4], xscale=log2, yscale=log10, xlabel="cells per wavelength", ylabel="|σ / σ_analytic − 1|", title="(d) convergence and the couplings")
for (pat, lab, mk, col) in ((r"^dyn_k0.6(_n\d+)?$", "dynamical, k = 0.6", :circle, :crimson), (r"^sec_ts1_k0.6(_n\d+)?$", "secular (no BR), k = 0.6", :utriangle, :seagreen),
                            (r"^br_f0.55_ts1_k0.6(_n\d+)?$", "near threshold, 4πGρ₀ = 0.55", :dtriangle, :purple))
    i = pick(pat); isempty(i) && continue; o = sortperm(nx1[i]); i = i[o]
    scatterlines!(ax4, nx1[i], relerr.(i), marker=mk, markersize=12, color=col, label=lab)
end
for (nm, lab, mk) in (("dyn_k0.6_fft", "FFT solver", :xcross), ("dyn_k0.6_np4", "4 ranks", :rect), ("dyn_k0.6_ts0.1", "t_s = 0.1", :star5),
                      ("dyn_k0.6_ts10", "t_s = 10", :pentagon), ("dyn_3sp_k0.6", "three species", :hexagon),
                      ("sec_ts1_k0.6_fft", "secular, FFT", :xcross), ("sec_ts1_k0.6_np4", "secular, 4 ranks", :rect), ("br_f0.6_ts1_k0.6_fft", "back-reaction, FFT", :xcross))
    i = first_(Regex("^" * nm * raw"$")); i === nothing && continue
    scatter!(ax4, [nx1[i]], [relerr(i)], marker=mk, markersize=14, label=lab)
end
lines!(ax4, [32, 128], 5e-2 .* (32 ./ [32, 128]).^2, color=:black, linestyle=:dash, label="2nd order")
axislegend(ax4, position=:lb, framevisible=false, labelsize=10)
save(joinpath(HERE, "dust4c_jeans_rot.png"), fig, px_per_unit=2)
println("wrote dust4c_jeans_rot.png")
