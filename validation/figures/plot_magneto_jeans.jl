# Magneto-Jeans test (ideal MHD + multigrid self-gravity), 2026-10-26: measured w^2 against
# the dispersion relation (w^2 - W_J^2)(w^2 - k^2 vA^2 cos^2 th) = k^2 vA^2 sin^2 th w^2.
# (a) w^2/(k cs)^2 against lambda/lambda_J for th = 0, 30, 60, 90 degrees (vA = cs; slow
#     branch dashed, fast solid; 64 cells per wavelength); open diamonds: the 3-D oblique wave
#     along the box diagonal (its own k and lambda/lambda_J), on the thin th = 45 and 90 curves;
# (b) th = 90, lambda/lambda_J = 1.5 against vA/cs: the field stabilizes the hydro-unstable
#     wave once k^2 vA^2 > 4 pi G rho0 - k^2 cs^2;
# (c) relative error of w against cells per wavelength (th = 60);
# (d) the perturbed field per perturbed density, dB_perp/drho, measured against the
#     eigenvector C = B_perp w^2/(rho0 (w^2 - k^2 vA_par^2)).
# Data: validation/run/magneto_jeans/results.tsv (magneto_jeans.py).
# Usage: julia validation/figures/plot_magneto_jeans.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
tab, hdr = readdlm(joinpath(dirname(@__DIR__), "run", "magneto_jeans", "results.tsv"), '\t',
                   header=true)
R = [Dict(String(h) => tab[i, j] for (j, h) in enumerate(vec(hdr))) for i in 1:size(tab, 1)]
K = 2π
function w2(n, vA, th, mode)
    t = deg2rad(th); wj2 = K^2*(1 - n^2)
    abs(sin(t)) < 1e-12 && return wj2
    abs(cos(t)) < 1e-12 && return wj2 + K^2*vA^2
    s = wj2 + K^2*vA^2; p = wj2*K^2*vA^2*cos(t)^2; d = sqrt(max(s^2 - 4p, 0.0))
    mode == "fast" ? (s + d)/2 : (s - d)/2
end
COL = Dict(0 => "#000000", 30 => "#3182bd", 45 => "#756bb1", 60 => "#e6550d", 90 => "#31a354")
sel(f) = filter(f, R)

fig = Figure(size=(1400, 1050))
a = Axis(fig[1, 1], xlabel="λ / λ_J", ylabel="ω² / (k c_s)²",
         title="(a) dispersion relation, v_A = c_s", titlealign=:left)
nn = range(0.3, 2.2, length=300)
for th in (0, 30, 60, 90)
    modes = th in (30, 60) ? ("slow", "fast") : ("fast",)
    for m in modes
        lines!(a, nn, [w2(n, 1.0, th, m) for n in nn] ./ K^2, color=COL[th],
               linestyle=(m == "slow" ? :dash : :solid), linewidth=1.5,
               label=(m == "fast" ? "θ = $(th)°" : nothing))
    end
    pts = sel(c -> c["dir"] == "x1" && c["th"] == th && c["vA"] == 1.0 && c["nx"] == 64)
    scatter!(a, [c["n"] for c in pts], [c["w2_meas"] for c in pts] ./ K^2, color=COL[th],
             markersize=11)
end
for m in ("slow", "fast")
    lines!(a, nn, [w2(n, 1.0, 45, m) for n in nn] ./ K^2, color=COL[45], linewidth=0.8,
           linestyle=(m == "slow" ? :dash : :solid))
end
for c in sel(c -> c["dir"] == "oblique")
    scatter!(a, [c["kj"]/c["kw"]], [c["w2_meas"]/c["kw"]^2], color=:transparent,
             strokecolor=COL[c["th"]], strokewidth=2, markersize=16, marker=:diamond)
end
hlines!(a, [0.0], color=:gray50, linewidth=0.8)
axislegend(a, position=:lb, framevisible=false, labelsize=12)
text!(a, 1.42, 1.6, align=(:left, :center), text="dashed: slow branch\n◇: 3-D oblique wave (θ = 45°: purple, thin)",
      fontsize=12, color=:gray30)

b = Axis(fig[1, 2], xlabel="v_A / c_s", ylabel="ω² / (k c_s)²",
         title="(b) θ = 90°, λ/λ_J = 1.5: magnetic support", titlealign=:left)
vv = range(0, 2.2, length=200)
lines!(b, vv, [w2(1.5, v, 90, "fast") for v in vv] ./ K^2, color=COL[90], linewidth=1.5)
pts = sel(c -> c["dir"] == "x1" && c["th"] == 90 && c["n"] == 1.5 && c["nx"] == 64)
scatter!(b, [c["vA"] for c in pts], [c["w2_meas"] for c in pts] ./ K^2, color=COL[90],
         markersize=11)
hlines!(b, [0.0], color=:gray50, linewidth=0.8)
vlines!(b, [sqrt(1.5^2 - 1)], color=:gray50, linestyle=:dot)
text!(b, sqrt(1.25) + 0.05, -1.0, text="v_A = c_s √(n² − 1)", fontsize=12, color=:gray30)

c3 = Axis(fig[2, 1], xlabel="cells per wavelength", ylabel="|Δω| / |ω|", xscale=log2,
          yscale=log10, title="(c) convergence, θ = 60°", titlealign=:left)
for (m, n, lab, mk) in (("slow", 2.0, "slow, λ/λ_J = 2 (growing)", :circle),
                       ("fast", 0.5, "fast, λ/λ_J = 0.5 (oscillating)", :rect))
    local pts = sort(sel(c -> c["dir"] == "x1" && c["th"] == 60 && c["mode"] == m && c["n"] == n),
               by=c -> c["nx"])
    x = [c["nx"] for c in pts]; y = [c["rel_err"] for c in pts]
    scatterlines!(c3, x, y, marker=mk, markersize=11, label=lab)
end
xr = [16, 128]
lines!(c3, xr, 3e-3 .* (16 ./ xr).^2, color=:gray50, linestyle=:dash, label="∝ N⁻²")
axislegend(c3, position=:lb, framevisible=false, labelsize=12)

d = Axis(fig[2, 2], xlabel="C = δB⊥/δρ (eigenvector)", ylabel="measured δB⊥/δρ",
         title="(d) perturbed field per perturbed density", titlealign=:left)
pts = sel(c -> abs(c["C_ana"]) > 0)
scatter!(d, [c["C_ana"] for c in pts], [c["C_meas"] for c in pts],
         color=[COL[c["th"]] for c in pts], markersize=10)
lo, hi = extrema([c["C_ana"] for c in pts]); pad = 0.05*(hi - lo)
lines!(d, [lo - pad, hi + pad], [lo - pad, hi + pad], color=:gray50, linestyle=:dash)
for th in (30, 45, 60, 90)
    scatter!(d, [NaN], [NaN], color=COL[th], label="θ = $(th)°")
end
axislegend(d, position=:lt, framevisible=false, labelsize=12)
Label(fig[0, 1:2], "Magneto-Jeans test: ideal MHD + multigrid self-gravity (c_s = ρ₀ = 1, λ = 1)",
      fontsize=18)
out = joinpath(@__DIR__, "magneto_jeans.png")
save(out, fig, px_per_unit=1.5)
println("wrote ", out)
