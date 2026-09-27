# Baehr+22 box, stage 1 (gas only): 40 cells/H_g on GPUs (gtb40_stage1, hse_outflow z
# faces) against the 20 cells/H_g CPU run (gt_baehr_stage1, diode).  Top: Sigma_g at
# t = 150 on one colour scale.  Bottom: alpha' (SC14 eq. 20) and Q against time from the
# user history, with the t = 75-150 saturated means.
# The 40/H colden00150.npz is written by the streaming reader of the 10.7 GB dump
# (scratch gtb40_sigma.py: sig[y, x], same layout as gt_baehr_stage1/colden.py).
# Usage: julia validation/figures/plot_gtb40_vs_gtb20.jl
using CairoMakie, NPZ, Printf, Statistics, DelimitedFiles
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
R = joinpath(dirname(@__DIR__), "run"); Hg = 2.16878; L = 55.2302; G = 4.25231/(4π)

# user history, keeping the last write of each time (continuations rewrite the tail)
function hist(dir)
    f = first(filter(p -> endswith(p, ".user.hst"), readdir(joinpath(R, dir), join=true)))
    d = readdlm(f, comments=true, comment_char='#')
    keep = falses(size(d, 1)); tmin = Inf
    for i in size(d, 1):-1:1
        d[i, 1] < tmin && (keep[i] = true; tmin = d[i, 1])
    end
    d = d[keep, :]
    t = d[:, 1]; M = d[:, 3]
    (t=t, ap=(2/3) .* (d[:, 8] .+ d[:, 9]) ./ d[:, 10],
     Q=(d[:, 4] ./ M) ./ (π*G .* M ./ L^2))
end

runs = [("gt_baehr_stage1", "20 cells/H_g  (CPU, diode)", :royalblue),
        ("gtb40_stage1",    "40 cells/H_g  (GPU, hse_outflow)", :crimson)]

fig = Figure(size=(1300, 1060))
for (j, (dir, lab, _)) in enumerate(runs)
    d = npzread(joinpath(R, dir, "colden00150.npz"))
    sig = d["sig"]; t = d["t"]
    x = range(d["x1min"]/Hg, d["x1max"]/Hg, length=size(sig, 2))
    y = range(d["x2min"]/Hg, d["x2max"]/Hg, length=size(sig, 1))
    ax = Axis(fig[1, j], xlabel="x / H_g", ylabel=j == 1 ? "y / H_g" : "", aspect=DataAspect(),
              title=@sprintf("%s\nΣ_g at Ωt = %.0f   (max/mean %.2f, rms δΣ/Σ %.2f)",
                             lab, t, maximum(sig)/mean(sig), std(sig)/mean(sig)))
    hm = heatmap!(ax, x, y, permutedims(sig), colormap=:magma, colorrange=(0.6, 6.3))
    j == 2 && Colorbar(fig[1, 3], hm, label="Σ_g")
end

axa = Axis(fig[2, 1:2], ylabel="α′  (eq. 20)", xticklabelsvisible=false)
axq = Axis(fig[3, 1:2], xlabel="Ωt", ylabel="Q")
for (dir, lab, c) in runs
    h = hist(dir); m = 75 .<= h.t .<= 150
    lines!(axa, h.t, h.ap, color=(c, 0.8), linewidth=1.2,
           label=@sprintf("%s: ⟨α′⟩ = %.4f", lab, mean(h.ap[m])))
    lines!(axq, h.t, h.Q, color=c, linewidth=1.8,
           label=@sprintf("%s: ⟨Q⟩ = %.3f", lab, mean(h.Q[m])))
end
vspan!(axa, 75, 150, color=(:gray, 0.12)); vspan!(axq, 75, 150, color=(:gray, 0.12))
linkxaxes!(axa, axq); xlims!(axq, 0, 150)
axislegend(axa, position=:lt, framevisible=false, labelsize=13)
axislegend(axq, position=:rb, framevisible=false, labelsize=13)
rowsize!(fig.layout, 1, Relative(0.58))
save(joinpath(@__DIR__, "gtb40_vs_gtb20.png"), fig, px_per_unit=2)
println("ok")
