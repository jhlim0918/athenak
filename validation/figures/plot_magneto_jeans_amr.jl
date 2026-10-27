# Magneto-Jeans test on refined meshes (ideal MHD + multigrid self-gravity), 2026-10-26.
# The eigenmodes of the uniform suite (plot_magneto_jeans.jl) on uniform 64 and 128 cells
# per wavelength (the bracket), a static refined slab at level 1 and 2, and a moving
# refined slab (swept through the box twice per run) at level 1 and 2; and the 3-D
# diagonal wave on uniform 64, a static central block, and the moving slab.
# (a) relative error in omega (or the growth rate) per mode and mesh;
# (b) the spurious cos(k xi) density component, max|a_cos| / max|a_sin| -- reflection and
#     phase error at the level interfaces (zero by symmetry on a uniform grid);
# (c) the largest relative divergence of B over the run, int|div B| dx dV / int|B| dV;
# (d) the measured density amplitude minus theory for the oscillating fast mode
#     (theta = 60, lambda/lambda_J = 0.5) on each mesh, and the MeshBlock count of the
#     moving-slab runs.
# Data: validation/run/magneto_jeans_amr/results.tsv and runs/*/MagJeans.user.hst
# (magneto_jeans_amr.py).
# Usage: julia validation/figures/plot_magneto_jeans_amr.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
D = joinpath(dirname(@__DIR__), "run", "magneto_jeans_amr")
tab, hdr = readdlm(joinpath(D, "results.tsv"), '\t', header=true)
R = [Dict(String(h) => tab[i, j] for (j, h) in enumerate(vec(hdr))) for i in 1:size(tab, 1)]

MESH = ["uni64", "uni128", "smr1", "smr2", "amr1", "amr2"]
MLAB = Dict("uni64" => "uniform 64", "uni128" => "uniform 128", "smr1" => "static, level 1",
            "smr2" => "static, level 2", "amr1" => "moving, level 1",
            "amr2" => "moving, level 2")
MCOL = Dict("uni64" => :black, "uni128" => :gray55, "smr1" => "#3182bd",
            "smr2" => "#08519c", "amr1" => "#e6550d", "amr2" => "#a63603")
MMRK = Dict("uni64" => :circle, "uni128" => :circle, "smr1" => :rect, "smr2" => :rect,
            "amr1" => :utriangle, "amr2" => :utriangle)
# modes along the x axis
MODES = [("x1", 60, "slow", 2.0), ("x1", 60, "fast", 0.5), ("x1", 90, "fast", 1.5),
         ("x1", 30, "fast", 2.0), ("oblique", 45, "slow", 2.0), ("oblique", 45, "fast", 0.8)]
MODELAB = ["θ=60° slow\nλ/λ_J=2", "θ=60° fast\nλ/λ_J=0.5", "θ=90°, v_A=1.5\nλ/λ_J=1.5",
           "θ=30° fast\nλ/λ_J=2", "3-D θ=45° slow\nλ/λ_J=2", "3-D θ=45° fast\nλ/λ_J=0.8"]
getc(dir, th, mode, n, mesh) = (r = filter(c -> c["dir"] == dir && c["th"] == th &&
    c["mode"] == mode && c["n"] == n && c["mesh"] == mesh, R); isempty(r) ? nothing : r[1])
offs = Dict(m => (k - 3.5)*0.12 for (k, m) in enumerate(MESH))

fig = Figure(size=(1500, 1100))
function panel(pos, key, ylab, title; floor=0.0)
    ax = Axis(pos, ylabel=ylab, yscale=log10, title=title, titlealign=:left,
              xticks=(1:length(MODES), MODELAB), xticklabelsize=12)
    for m in MESH
        xs = Float64[]; ys = Float64[]
        for (k, (dir, th, mode, n)) in enumerate(MODES)
            c = getc(dir, th, mode, n, m)
            c === nothing && continue
            push!(xs, k + offs[m]); push!(ys, max(Float64(c[key]), floor))
        end
        isempty(xs) && continue
        scatter!(ax, xs, ys, color=MCOL[m], marker=MMRK[m], markersize=13, label=MLAB[m])
    end
    ax
end
a = panel(fig[1, 1], "rel_err", "|Δω| / |ω|", "(a) error in the frequency or growth rate")
axislegend(a, position=:lt, framevisible=false, labelsize=12, nbanks=2)
b = panel(fig[1, 2], "acos_rel_max", "max|a_cos| / max|a_sin|",
          "(b) spurious cos(kξ) density (interface reflection, phase)", floor=1e-16)
c = panel(fig[2, 1], "divb_rel_max", "max ∫|∇·B|Δx dV / ∫|B| dV",
          "(c) divergence of B", floor=1e-18)

# (d) residual of the oscillating fast mode, and the block count of the moving runs
d = Axis(fig[2, 2], xlabel="t", ylabel="a(t)/a₀ − cos ω t",
         title="(d) θ = 60° fast, λ/λ_J = 0.5: measured minus theory", titlealign=:left)
d2 = Axis(fig[2, 2], ylabel="MeshBlocks (moving runs)", yaxisposition=:right,
          ylabelcolor=:gray40, yticklabelcolor=:gray40)
hidespines!(d2); hidexdecorations!(d2)
w = sqrt(Float64(getc("x1", 60, "fast", 0.5, "uni64")["w2_ana"]))
for m in MESH
    f = joinpath(D, "runs", "x1_th60_fast_n0.5_vA1.0_$(m)", "MagJeans.user.hst")
    isfile(f) || continue
    h = readdlm(f, comments=true)
    t = h[:, 1]; a0 = h[1, 3]
    lines!(d, t, h[:, 3] ./ a0 .- cos.(w .* t), color=MCOL[m], linewidth=1.5, label=MLAB[m])
    if startswith(m, "amr")
        lines!(d2, t, h[:, 10], color=MCOL[m], linestyle=:dot, linewidth=1.2)
    end
end
hlines!(d, [0.0], color=:gray60, linewidth=0.8)
linkxaxes!(d, d2)
Label(fig[0, 1:2], "Magneto-Jeans test on refined meshes: ideal MHD + multigrid self-gravity " *
      "(root 64 cells per wavelength)", fontsize=18)
out = joinpath(@__DIR__, "magneto_jeans_amr.png")
save(out, fig, px_per_unit=1.5)
println("wrote ", out)
