# Hydrodynamic Jeans test (hydro + multigrid self-gravity), 2026-11-03: measured w^2 against
# w^2 = k^2 cs^2 - 4 pi G rho0, the hydro counterpart of plot_magneto_jeans.jl.
# (a) w^2/(k cs)^2 against lambda/lambda_J: the line 1 - (lambda/lambda_J)^2; adiabatic and
#     isothermal runs at 64 cells per wavelength, the 3-D wave along the box diagonal;
# (b) relative error of w (or of the growth rate s) against lambda/lambda_J, measured against
#     the continuum relation and against the discrete one (7-point Laplacian eigenvalue and
#     the centred gravity gradient), which removes the rise near marginal stability;
# (c) relative error against cells per wavelength, lambda/lambda_J = 0.5 and 2;
# (d) refined meshes: the error on uniform 64 / 128 cells per wavelength (the bracket), static
#     slabs (1 and 2 levels), the moving AMR window (1 and 2 levels), and the 3-D runs.
# Data: validation/run/hydro_jeans/results.tsv (hydro_jeans.py).
# Usage: julia validation/figures/plot_hydro_jeans.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
tab, hdr = readdlm(joinpath(dirname(@__DIR__), "run", "hydro_jeans", "results.tsv"), '\t',
                   header=true)
R = [Dict(String(h) => tab[i, j] for (j, h) in enumerate(vec(hdr))) for i in 1:size(tab, 1)]
sel(f) = filter(f, R)
cA, cI, c3 = "#000000", "#e6550d", "#3182bd"

fig = Figure(size=(1400, 1050))

# (a) dispersion relation
a = Axis(fig[1, 1], xlabel="λ / λ_J", ylabel="ω² / (k c_s)²",
         title="(a) dispersion relation (ω² < 0: growth rate s² = −ω²)", titlealign=:left)
nn = range(0.0, 3.2, length=300)
lines!(a, nn, 1 .- nn.^2, color=:gray55, linewidth=2, label="1 − (λ/λ_J)²")
hlines!(a, [0.0], color=:gray80, linestyle=:dot)
for (f, col, mk, lab) in ((r -> r["tag"] in ("disp", "res") && r["nx"] == 64 && r["mesh"] == "uni",
                           cA, :circle, "adiabatic, 64 cells per λ"),
                          (r -> r["tag"] == "iso", cI, :utriangle, "isothermal, 64 cells per λ"),
                          (r -> r["tag"] == "obl", c3, :diamond, "3-D, box diagonal"))
    s = sel(f)
    isempty(s) && continue
    scatter!(a, [r["n"] for r in s], [r["w2_meas"]/r["k2"] for r in s], color=col, marker=mk,
             markersize=11, strokewidth=(mk == :circle ? 0 : 1), label=lab)
end
axislegend(a, position=:lb, framevisible=false)

# (b) error against lambda/lambda_J
b = Axis(fig[1, 2], xlabel="λ / λ_J", ylabel="relative error in ω (or s)", yscale=log10,
         title="(b) error, 64 cells per λ (adiabatic)", titlealign=:left)
s = sort(sel(r -> r["tag"] == "disp"), by=r -> r["n"])
ns = [r["n"] for r in s]
scatterlines!(b, ns, [max(r["err_cont"], 1e-7) for r in s], color=cA, marker=:circle,
              label="vs continuum")
scatterlines!(b, ns, [max(r["err_disc"], 1e-7) for r in s], color=cI, marker=:rect,
              linestyle=:dash, label="vs discrete operators")
vlines!(b, [1.0], color=:gray80, linestyle=:dot)
axislegend(b, position=:rt, framevisible=false)

# (c) convergence
c = Axis(fig[2, 1], xlabel="cells per wavelength", ylabel="relative error in ω (or s)",
         xscale=log2, yscale=log10, title="(c) convergence", titlealign=:left)
for (n, col, lab) in ((0.5, cA, "λ/λ_J = 0.5 (oscillating)"), (2.0, cI, "λ/λ_J = 2 (growing)"))
    sc = sort(sel(r -> r["tag"] in ("disp", "res") && r["n"] == n && r["mesh"] == "uni" &&
                       r["dir"] == "x1"), by=r -> r["nx"])
    x = [r["nx"] for r in sc]
    scatterlines!(c, x, [r["err_cont"] for r in sc], color=col, marker=:circle, label=lab)
    scatterlines!(c, x, [r["err_disc"] for r in sc], color=col, marker=:rect, linestyle=:dash)
end
x0 = [16, 128]
lines!(c, x0, 2e-2 .* (x0 ./ 16).^-2, color=:gray60, linestyle=:dot, label="2nd order")
axislegend(c, position=:lb, framevisible=false)
text!(c, 0.98, 0.98, text="solid: vs continuum, dashed: vs discrete", space=:relative,
      align=(:right, :top), fontsize=12)

# (d) refined meshes
d = Axis(fig[2, 2], ylabel="relative error in ω (or s), vs continuum", yscale=log10,
         title="(d) refined meshes (1-D: 64 cells per λ root; 3-D: 21)", titlealign=:left)
meshes = ["uni64", "uni128", "smr1", "smr2", "amr1", "amr2"]
function meshrun(n, mesh, dir)
    if mesh == "uni64"
        f = r -> r["mesh"] == "uni" && r["nx"] == 64 && r["n"] == n && r["dir"] == dir &&
                 r["eos"] == "ideal"
    elseif mesh == "uni128"
        f = r -> r["mesh"] == "uni" && r["nx"] == 128 && r["n"] == n && r["dir"] == dir
    else
        f = r -> r["mesh"] == mesh && r["n"] == n && r["dir"] == dir
    end
    s = sel(f)
    isempty(s) ? nothing : s[1]
end
for (k, (n, dir, col, mk, lab)) in enumerate(((0.5, "x1", cA, :circle, "λ/λ_J = 0.5"),
                                              (2.0, "x1", cI, :utriangle, "λ/λ_J = 2"),
                                              (0.5, "oblique", c3, :diamond, "3-D, λ/λ_J = 0.5"),
                                              (2.0, "oblique", "#08519c", :rect, "3-D, λ/λ_J = 2")))
    xs, ys = Int[], Float64[]
    for (i, m) in enumerate(meshes)
        r = meshrun(n, m, dir)
        r === nothing && continue
        push!(xs, i); push!(ys, r["err_cont"])
    end
    scatter!(d, xs .+ 0.12*(k - 2.5), ys, color=col, marker=mk, markersize=12, label=lab)
end
d.xticks = (1:6, meshes)
axislegend(d, position=(:center, :bottom), framevisible=false, nbanks=2)

Label(fig[0, 1:2], "Hydrodynamic Jeans test: hydro + multigrid self-gravity (c_s = ρ₀ = 1, λ = 1)",
      fontsize=18)
out = joinpath(@__DIR__, "hydro_jeans.png")
save(out, fig, px_per_unit=1.5)
println("wrote ", out)
