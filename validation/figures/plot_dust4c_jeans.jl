# Dusty Jeans instability with superparticle dust (Krapp et al. 2024 sec. 3.5): measured
# growth rates against the linear eigenproblem.  Left: sigma(k/kJ) for eps = 1, taus = 1
# (analytic: dust with back-reaction, without, and the eps = 0 gas curve) with the gas-mode
# rates measured at 16, 32, 64 and 128 cells per wavelength.  Middle: the replica of Krapp's
# fig. 6 -- 128 species, eps0 = 0.01, taus log-uniform in [1e-4, 10] (what their plotted
# curve is), sigma in their units (times sqrt(1 + eps0)), their six wavenumbers at 16 / 32 /
# 64 / 128 cells per wavelength.  Right: convergence at fixed k for the eps = 1 case, the
# linear-taus 128-species mixture and the replica's cutoff mode, and the other couplings and
# solvers.  Data: validation/run/dust4c/jeans/ (jeans_results.txt from battery.sh, curve_*.txt
# from scripts/analysis/dust_jeans.py curve).  Usage: julia validation/figures/plot_dust4c_jeans.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run", "dust4c", "jeans")

# results table: name k nx1 nspec sigma_analytic sigma_gas_measured sigma_dust1_measured
tab = readdlm(joinpath(RUN, "jeans_results.txt"), comments=true)
name = String.(tab[:, 1]); k = Float64.(tab[:, 2]); nx1 = Int.(tab[:, 3]); nsp = Int.(tab[:, 4])
sana = Float64.(tab[:, 5]); sgas = Float64.(tab[:, 6]); sdust = Float64.(tab[:, 7])
c1  = readdlm(joinpath(RUN, "curve_e1_ts1.txt"), comments=true)
cl128 = readdlm(joinpath(RUN, "curve_l128.txt"), comments=true)
relerr(i) = abs(sgas[i]/sana[i] - 1) + 1e-16

cols = Dict(16 => :dodgerblue, 32 => :darkorange, 64 => :green3, 128 => :red)
function respoints!(ax, sel, scale=1.0; ns=[16, 32, 64, 128], lab=n -> "measured, $n cells/λ")
    for n in ns
        s = sel .& (nx1 .== n); any(s) || continue
        scatter!(ax, k[s], scale .* sgas[s], marker=:circle, markersize=13, color=(:white, 0), strokecolor=cols[n], strokewidth=1.8,
                 label=lab(n))
    end
end

fig = Figure(size=(1750, 560))
# --- left: eps = 1, taus = 1
ax1 = Axis(fig[1, 1], xlabel="k / k_J", ylabel="σ  (c_s k_J)", title="ε₀ = 1, t_s = 1/(c_s k_J), one species")
lines!(ax1, c1[:, 1], c1[:, 2], linewidth=2, label="analytic, dust + back-reaction")
lines!(ax1, c1[:, 1], c1[:, 3], linewidth=2, linestyle=:dash, label="analytic, no back-reaction")
lines!(ax1, c1[:, 1], c1[:, 4], linewidth=2, linestyle=:dot, color=:gray, label="gas only (ε₀ = 0)")
sel1 = [occursin(r"^e1_k[0-9.]+(_n\d+)?$", n) for n in name] .& (nsp .== 1)
respoints!(ax1, sel1)
nb = findfirst(==("e1_k06_nobr"), name)
nb === nothing || scatter!(ax1, [k[nb]], [sgas[nb]], markersize=14, marker=:diamond, color=:orange, label="no back-reaction, 128 cells/λ")
axislegend(ax1, position=:lb, framevisible=false, labelsize=12)
ylims!(ax1, 0, 1.05)

# --- middle: the replica of Krapp's fig. 6 (their sigma = Re(s) sqrt(1 + eps0), k up to 1.5)
q = sqrt(1.01)
ax2 = Axis(fig[1, 2], xlabel="k / k_J", ylabel="σ", title="Krapp+24 fig. 6: 128 species, ε⁰ = 0.01, t_s = 10⁻⁴ – 10 (log-uniform)")
lines!(ax2, cl128[:, 1], q .* cl128[:, 4], linewidth=1.5, linestyle=:dash, color=:black, label="Gas")
lines!(ax2, cl128[:, 1], q .* cl128[:, 2], linewidth=1.5, color=:black, label="128 fluids + gas")
vlines!(ax2, [1/q], color=:black, linestyle=:dot)
respoints!(ax2, [startswith(nm, "l128_") for nm in name], q; lab=n -> "N_cells = $n")
axislegend(ax2, position=:rt, framevisible=false, labelsize=12)
text!(ax2, 0.05, 0.08, text="t_s = 10⁻⁴ – 10,  ε = 0.01", fontsize=13)
xlims!(ax2, 0, 1.5); ylims!(ax2, 0, 1.05)

# --- right: convergence at k = 0.6 and the code variants
ax3 = Axis(fig[1, 3], xscale=log2, xlabel="cells per wavelength", ylabel="|σ / σ_analytic − 1|", yscale=log10,
           title="convergence at fixed k, and the couplings (k = 0.6)")
r1 = [i for i in eachindex(name) if occursin(r"^e1_k0?\.?6(_n\d+)?$", name[i]) && nsp[i] == 1]
o = sortperm(nx1[r1]); r1 = r1[o]
scatterlines!(ax3, nx1[r1], relerr.(r1), markersize=12, label="ε₀ = 1, one species (IMEX, multigrid)")
r2 = [i for i in eachindex(name) if occursin(r"^n128_k0\.6_n\d+$", name[i])]
o = sortperm(nx1[r2]); r2 = r2[o]
isempty(r2) || scatterlines!(ax3, nx1[r2], relerr.(r2), markersize=12, marker=:utriangle, label="ε₀ = 0.01, 128 species (t_s linear), k = 0.6")
r3 = [i for i in eachindex(name) if occursin(r"^l128_k0\.995_n\d+$", name[i])]
o = sortperm(nx1[r3]); r3 = r3[o]
isempty(r3) || scatterlines!(ax3, nx1[r3], relerr.(r3), markersize=12, marker=:dtriangle, label="128 species (t_s log), k = 0.995 (gas cutoff)")
for (tag, lab, mk) in (("e1_k06_pc2", "PC2/rk2", :diamond), ("e1_k06_fft", "FFT solver", :xcross), ("e1_k06_np4", "4 ranks", :rect),
                       ("e1_ts001_k06", "t_s = 0.01 (IMEX)", :star5), ("e1_ts100_k06", "t_s = 100", :pentagon),
                       ("e1_3sp_k06", "three species", :hexagon))
    i = findfirst(==(tag), name); i === nothing && continue
    scatter!(ax3, [nx1[i]], [relerr(i)], markersize=14, marker=mk, label=lab)
end
lines!(ax3, [16, 256], 3e-2 .* (16 ./ [16, 256]).^2, color=:black, linestyle=:dash, label="2nd order")
axislegend(ax3, position=:rt, framevisible=false, labelsize=11)
save(joinpath(HERE, "dust4c_jeans.png"), fig, px_per_unit=2)
println("wrote dust4c_jeans.png")
