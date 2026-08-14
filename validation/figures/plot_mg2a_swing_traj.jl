# Swing-amplification trajectories for the Phase-2a A/B (doc fig, sec. 10.5):
# d_cos/A vs time for ring-SMR MG, uniform MG, uniform FFT, plus the pairwise
# trajectory differences — the uniform MG-vs-FFT gap (1e-7, solver seam) sits five
# decades below the SMR-vs-uniform gap (5e-2, hydro resolution blend).
# Data: validation/run/sw2a_*.user.hst (notebook sec. 12 runs).
using CairoMakie, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const HERE = @__DIR__
const VAL = dirname(HERE)
const RUN = joinpath(VAL, "run")
const A = 1e-4

function read_hst(f)
    rows = Float64[]; ncol = 0
    for ln in eachline(f)
        startswith(strip(ln), "#") && continue
        v = parse.(Float64, split(ln)); ncol = length(v); append!(rows, v)
    end
    permutedims(reshape(rows, ncol, :))
end
H(bn) = read_hst(joinpath(RUN, bn * ".user.hst"))
hs, hu, hf = H("sw2a_smr"), H("sw2a_unif_mg"), H("sw2a_unif_fft")

fig = Figure(size=(900, 640))
ax1 = Axis(fig[1, 1]; ylabel="d_cos / A", title="swing amplification: level-1 ring vs uniform")
lines!(ax1, hs[:, 1], hs[:, 3] ./ A; color=:crimson, linewidth=2.2, label="ring SMR, MG")
lines!(ax1, hu[:, 1], hu[:, 3] ./ A; color=:steelblue, linewidth=1.4, label="uniform, MG")
lines!(ax1, hf[:, 1], hf[:, 3] ./ A; color=:black, linewidth=1.0, linestyle=:dash,
       label="uniform, FFT")
ipk = argmax(abs.(hs[:, 3]))
scatter!(ax1, [hs[ipk, 1]], [hs[ipk, 3] / A]; color=:crimson, marker=:diamond)
text!(ax1, hs[ipk, 1] - 0.15, hs[ipk, 3] / A;
      text=@sprintf("%.4f (SMR)\n%.4f (uniform)", hs[ipk, 3] / A, hu[ipk, 3] / A),
      align=(:right, :center), fontsize=12)
axislegend(ax1; position=:lb, framevisible=false)

n = min(size(hs, 1), size(hu, 1), size(hf, 1))
ax2 = Axis(fig[2, 1]; xlabel="t", ylabel="|Δ d_cos| / A", yscale=log10)
lines!(ax2, hs[1:n, 1], max.(abs.(hs[1:n, 3] .- hu[1:n, 3]) ./ A, 1e-12);
       color=:crimson, label="SMR − uniform (hydro truncation)")
lines!(ax2, hs[1:n, 1], max.(abs.(hu[1:n, 3] .- hf[1:n, 3]) ./ A, 1e-12);
       color=:steelblue, label="uniform: MG − FFT (solver seam)")
axislegend(ax2; position=:rb, framevisible=false, labelsize=11)
rowsize!(fig.layout, 1, Relative(0.62))

save(joinpath(HERE, "mg2a_swing_traj.png"), fig)
println("wrote figures/mg2a_swing_traj.png")
@printf("peak: SMR %.6f  uniform %.6f  (both at t=%.3f); max diffs: SMR-unif %.2e, MG-FFT %.2e\n",
        hs[ipk, 3] / A, hu[ipk, 3] / A, hs[ipk, 1],
        maximum(abs.(hs[1:n, 3] .- hu[1:n, 3])) / A,
        maximum(abs.(hu[1:n, 3] .- hf[1:n, 3])) / A)
fig
