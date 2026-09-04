# Phase 4c: dust self-gravity.
# Left: the force gathered at the particles vs. the analytic acceleration of the TS23
# triple-sine potential (dust twin, NGP and TSC deposits) against resolution, with a
# second-order guide.  Middle: the vertical orbit of test particles in the isothermal
# self-gravitating slab (32x32x64, h = 9 cells, z0 = 0.21 h): <z>(t)/z0 at the default
# step with the exact anharmonic trajectory's turning points marked by the period.
# Right: the relative period error against the step, IMEX and PC2, with NGP for contrast.
#
# Data: validation/run/dust4c/ (README there)
# Usage: julia validation/figures/plot_dust4c_gravity.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))

const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run", "dust4c")

# --- (a) force error vs resolution, from twin_results.txt
lines = readlines(joinpath(RUN, "twin", "twin_results.txt"))
function ferr(tag)
    for l in lines
        startswith(l, tag * ".log: # DUST-GRAVITY FORCE ERROR") || continue
        return parse(Float64, match(r"max_rel= ([\d.e+-]+)", l).captures[1])
    end
    return NaN
end
N = [32, 64, 128]
e_ngp = [ferr("ngp_n32"), ferr("ngp_f1"), ferr("ngp_n128")]
e_tsc = [ferr("tsc_n32"), ferr("tsc_f1"), ferr("tsc_n128")]
println("NGP force errors ", e_ngp, "  orders ", round.(log2.(e_ngp[1:2] ./ e_ngp[2:3]), digits=2))
println("TSC force errors ", e_tsc, "  orders ", round.(log2.(e_tsc[1:2] ./ e_tsc[2:3]), digits=2))

fig = Figure(size=(1500, 440))
ax1 = Axis(fig[1, 1], xscale=log2, yscale=log10, xlabel="cells per side", ylabel="max relative force error",
           title="gathered force vs. analytic (sin3 twin)")
scatterlines!(ax1, N, e_ngp, label="NGP deposit/gather", markersize=12)
scatterlines!(ax1, N, e_tsc, label="TSC deposit/gather", markersize=12)
lines!(ax1, N, e_ngp[1] .* (N[1] ./ N).^2, color=:black, linestyle=:dash, label="2nd order")
axislegend(ax1, position=:lb, framevisible=false)

# --- (b) the orbit
hst(path) = readdlm(path, comments=true, comment_char='#')
u = hst(joinpath(RUN, "orbit", "dt_0.025.user.hst"))
z0 = 0.03; Texact = 6.318315
ax2 = Axis(fig[1, 2], xlabel="t", ylabel="⟨z⟩ / z0", title="test particles in the isothermal slab (dt = 0.025)")
lines!(ax2, u[:, 1], (u[:, 3] ./ u[:, 6]) ./ z0, linewidth=2, label="64 particles, mean")
vlines!(ax2, Texact .* (0.5:0.5:5), color=:gray, linestyle=:dot, label="exact half-periods")
axislegend(ax2, position=:rt, framevisible=false)

# --- (c) period error vs dt
function period(path)
    d = hst(path); t = d[:, 1]; vz = d[:, 4] ./ d[:, 6]
    s = sign.(vz); idx = findall(i -> s[i]*s[i+1] < 0, 1:length(s)-1)
    tc = [t[i] - vz[i]*(t[i+1]-t[i])/(vz[i+1]-vz[i]) for i in idx]
    per = vcat(diff(tc[1:2:end]), diff(tc[2:2:end]))
    return sum(per)/length(per), d[2, 2]
end
runs = [("dt_def", "IMEX"), ("dt_0.05", "IMEX"), ("dt_0.025", "IMEX"), ("dt_0.0125", "IMEX"), ("pc2_def", "PC2/rk2"), ("dt_def_ngp", "IMEX, NGP")]
dts = Float64[]; errs = Float64[]; labs = String[]
for (r, lab) in runs
    T, dt = period(joinpath(RUN, "orbit", r * ".user.hst"))
    push!(dts, dt); push!(errs, abs(T/Texact - 1)); push!(labs, lab)
    @printf("%-12s dt=%.5f T=%.6f  |T/Texact-1|=%.2e\n", r, dt, T, abs(T/Texact - 1))
end
ax3 = Axis(fig[1, 3], xscale=log10, yscale=log10, xlabel="Δt", ylabel="|T / T_exact − 1|",
           title="period error (exact period 6.31832; harmonic 6.28319)")
imex = labs .== "IMEX"
scatterlines!(ax3, dts[imex], errs[imex], label="IMEX (TSC)", markersize=12)
scatter!(ax3, dts[labs .== "PC2/rk2"], errs[labs .== "PC2/rk2"], label="PC2/rk2 (TSC)", markersize=14, marker=:diamond)
scatter!(ax3, dts[labs .== "IMEX, NGP"], errs[labs .== "IMEX, NGP"], label="IMEX (NGP)", markersize=14, marker=:utriangle)
hlines!(ax3, [abs(6.283185/Texact - 1)], color=:gray, linestyle=:dash, label="harmonic approximation")
axislegend(ax3, position=:lt, framevisible=false)

save(joinpath(HERE, "dust4c_gravity.png"), fig, px_per_unit=2)
println("wrote dust4c_gravity.png")
fig
