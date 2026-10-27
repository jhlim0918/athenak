# Magnetized swing amplification (ideal MHD + multigrid self-gravity + shear), 2026-10-26.
# A leading shearing wave (kx0 = -6, ky = 1) in a uniform azimuthal field B0 = b0 y_hat,
# adiabatic cs = 1, 4 pi G rho0 = 1.9; measured projections (points) against the linear
# system integrated with the 7-point Laplacian eigenvalue (lines; swing_ode.py, exported to
# the runs' times by swing_mhd.py as runs/<case>/theory.tsv).
# (a) density amplitude delta(t)/A on the uniform grid with orbital advection: b0 = 0, 0.5,
#     1.0, and b0 = 0.5 without self-gravity;
# (b) the azimuthal field perturbation c_y(t)/A for b0 = 0.5 and 1.0;
# (c) b0 = 1.0 without orbital advection: the field perturbation c_y on the uniform mesh, the
#     static level-1 ring and the moving ring (the refined runs' field error; shaded: swing);
# (d) the relative divergence of B, int|div B| dx dV / int|B| dV, and the moving-ring runs'
#     MeshBlock count.
# Usage: julia validation/figures/plot_swing_mhd.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
D = joinpath(dirname(@__DIR__), "run", "swing_mhd")
A = 1.0e-4
hst(c) = readdlm(joinpath(D, "runs", c, "SwingMHD.user.hst"), comments=true)
th(c) = readdlm(joinpath(D, "runs", c, "theory.tsv"), comments=true)   # t delta wx wy cx cy

fig = Figure(size=(1500, 1100))
a = Axis(fig[1, 1], xlabel="Ω t", ylabel="δ / A",
         title="(a) density, uniform 128², orbital advection", titlealign=:left)
for (c, col, lab) in (("uF_b0.0", :black, "B₀ = 0"), ("uF_b0.5", "#3182bd", "B₀ = 0.5 (β = 4.8)"),
                      ("uF_b1.0", "#e6550d", "B₀ = 1.0 (β = 1.2)"),
                      ("uF_b0.5_nograv", :gray55, "B₀ = 0.5, no gravity"))
    isfile(joinpath(D, "runs", c, "theory.tsv")) || continue
    h = hst(c); T = th(c)
    lines!(a, T[:, 1], T[:, 2], color=col, linewidth=1.6, label=lab)
    scatter!(a, h[1:10:end, 1], h[1:10:end, 3] ./ A, color=col, markersize=6)
end
vlines!(a, [4.0], color=:gray70, linestyle=:dot)
axislegend(a, position=:lb, framevisible=false, labelsize=12)

b = Axis(fig[1, 2], xlabel="Ω t", ylabel="c_y / A  (δB_y = c_y cos k·x)",
         title="(b) azimuthal field perturbation", titlealign=:left)
for (c, col, lab) in (("uF_b0.5", "#3182bd", "B₀ = 0.5"), ("uF_b1.0", "#e6550d", "B₀ = 1.0"),
                      ("uF_b0.5_256", "#08519c", "B₀ = 0.5, 256²"))
    isfile(joinpath(D, "runs", c, "theory.tsv")) || continue
    h = hst(c); T = th(c)
    lines!(b, T[:, 1], T[:, 6], color=col, linewidth=1.6, label=lab,
           linestyle=(endswith(c, "256") ? :dash : :solid))
    scatter!(b, h[1:10:end, 1], h[1:10:end, 9] ./ A, color=col, markersize=6)
end
axislegend(b, position=:lb, framevisible=false, labelsize=12)

c3 = Axis(fig[2, 1], xlabel="Ω t", ylabel="c_y / A",
          title="(c) B₀ = 1.0, no orbital advection: field perturbation on refined meshes",
          titlealign=:left)
T = isfile(joinpath(D, "runs", "uN_b1.0", "theory.tsv")) ? th("uN_b1.0") : nothing
T === nothing || lines!(c3, T[:, 1], T[:, 6], color=:black, linewidth=1.4, label="linear theory")
for (c, col, mk, lab) in (("uN_b1.0", :gray50, :circle, "uniform"),
                          ("smr_b1.0", "#3182bd", :rect, "static ring |x| < π/4"),
                          ("amr_b1.0", "#e6550d", :utriangle, "moving ring"))
    isfile(joinpath(D, "runs", c, "SwingMHD.user.hst")) || continue
    h = hst(c)
    scatter!(c3, h[1:6:end, 1], h[1:6:end, 9] ./ A, color=col, marker=mk, markersize=7,
             label=lab)
end
vspan!(c3, [3.0], [6.0], color=(:gray80, 0.3))
axislegend(c3, position=:lt, framevisible=false, labelsize=12)

d = Axis(fig[2, 2], xlabel="Ω t", ylabel="∫|∇·B|Δx dV / ∫|B| dV", yscale=log10,
         title="(d) divergence of B; MeshBlocks of the moving-ring runs", titlealign=:left)
d2 = Axis(fig[2, 2], ylabel="MeshBlocks", yaxisposition=:right, ylabelcolor=:gray40,
          yticklabelcolor=:gray40)
hidespines!(d2); hidexdecorations!(d2)
for (c, col) in (("uF_b0.5", "#3182bd"), ("uF_b1.0", "#e6550d"), ("uN_b0.5", :gray50),
                 ("smr_b0.5", "#08519c"), ("amr_b0.5", "#a63603"), ("amr_b1.0", "#fd8d3c"))
    isfile(joinpath(D, "runs", c, "SwingMHD.user.hst")) || continue
    h = hst(c)
    r = max.(h[2:end, 10] ./ h[2:end, 11], 1e-18)
    lines!(d, h[2:end, 1], r, color=col, linewidth=1.3, label=c)
    startswith(c, "amr") && lines!(d2, h[:, 1], h[:, 12], color=col, linestyle=:dot)
end
linkxaxes!(d, d2)
axislegend(d, position=:lt, framevisible=false, labelsize=11, nbanks=2)
Label(fig[0, 1:2], "Magnetized swing amplification: ideal MHD + multigrid self-gravity in the " *
      "shearing box (k_y = κ = 1, 4πGρ₀ = 1.9, c_s = 1)", fontsize=18)
out = joinpath(@__DIR__, "swing_mhd.png")
save(out, fig, px_per_unit=1.5)
println("wrote ", out)
