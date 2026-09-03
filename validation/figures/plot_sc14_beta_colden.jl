# Final vertically integrated gas density Sigma_g(x,y) = int rho dz across the
# standard-resolution SC14 beta scan (256 x 256 x 48, box 64H x 64H x 12H).
#
# One panel per run, last available hydro_w dump, shared logarithmic color scale
# in units of the initial surface density Sigma_0 = 2 rho_0 H = 2, so 0 dex is the
# unperturbed disk.  Each panel is annotated with its box-averaged <Sigma>/Sigma_0
# (mass retained) and peak Sigma/Sigma_0 (clump contrast).
#
# Usage: julia validation/figures/plot_sc14_beta_colden.jl
# Optional env: VMIN/VMAX (default -1.4 .. 1.2), PX supersampling (default 2).
using CairoMakie, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
include(joinpath(dirname(@__DIR__), "..", "scripts", "athenak_bin.jl"))

const HERE  = @__DIR__
const RUN   = joinpath(dirname(HERE), "run")
const SIG0  = 2.0

# beta, run directory, status (:ok / :hot non-steady / :bad numerically failed)
const RUNS = [
    (3.0,  "gt_sc14_stndrd_b3_morphing", :hot),
    (4.0,  "gt_sc14_stndrd_b4",          :ok),
    (5.0,  "gt_sc14_stndrd_b5",          :ok),
    (10.0, "gt_sc14_stndrd_b10",         :ok),
    (40.0, "gt_sc14_stndrd_b40",         :ok),
    (80.0, "gt_sc14_stndrd_b80",         :bad),
]

last_dump(dir) = last(sort(filter(f -> occursin("hydro_w", f) && endswith(f, ".bin"),
                                  readdir(joinpath(RUN, dir, "bin"), join=true))))

vmin = parse(Float64, get(ENV, "VMIN", "-1.4"))
vmax = parse(Float64, get(ENV, "VMAX",  "1.2"))
px   = parse(Float64, get(ENV, "PX", "2"))

fig = Figure(size=(1420, 980))
Label(fig[0, 1:3],
      "Final vertically integrated gas density Σ_g(x,y): SC14 β-scan, standard resolution (256×256×48)",
      fontsize=19, font=:bold, padding=(0, 0, 6, 0))

hm = nothing
for (n, (beta, dir, status)) in enumerate(RUNS)
    f   = last_dump(dir)
    fd  = read_bin(f)
    rho = assemble_root(fd, "dens")
    dz  = (fd.x3max - fd.x3min)/fd.Nx3
    sig = dropdims(sum(rho, dims=3), dims=3) .* dz ./ SIG0
    xs  = range(fd.x1min, fd.x1max, length=fd.Nx1)
    ys  = range(fd.x2min, fd.x2max, length=fd.Nx2)

    row, col = fldmod1(n, 3)
    tcol = status == :hot ? RGBf(0.55, 0.25, 0.65) :
           status == :bad ? RGBf(0.45, 0.45, 0.45) : :black
    note = status == :hot ? "  (morphed, non-steady)" :
           status == :bad ? "  (boundary runaway)" : ""
    ax = Axis(fig[row, col];
              title = @sprintf("β = %.0f,  Ωt = %.0f%s", beta, fd.time, note),
              titlecolor = tcol, titlesize = 17, titlefont = :bold,
              xlabel = row == 2 ? "x / H" : "", ylabel = col == 1 ? "y / H" : "",
              xticks = -32:16:32, yticks = -32:16:32,
              xticklabelsvisible = row == 2, yticklabelsvisible = col == 1,
              aspect = DataAspect())
    global hm = heatmap!(ax, xs, ys, log10.(clamp.(sig, 1e-10, Inf));
                  colormap = :inferno, colorrange = (vmin, vmax))
    xlims!(ax, fd.x1min, fd.x1max); ylims!(ax, fd.x2min, fd.x2max)

    lab = @sprintf("⟨Σ⟩ = %.2f Σ₀\nΣ_max = %.1f Σ₀", mean(sig), maximum(sig))
    x0, y0 = fd.x1min + 1.2, fd.x2min + 1.2
    poly!(ax, Rect2f(x0, y0, 22.0, 7.6); color = (:black, 0.55),
          strokecolor = (:white, 0.25), strokewidth = 0.8)
    text!(ax, x0 + 1.0, y0 + 1.0; text = lab, align = (:left, :bottom),
          color = :white, fontsize = 13)

    @printf("beta=%-4.0f t=%6.1f  <Sigma>/S0=%6.3f  max=%7.2f  min=%7.4f  (%s)\n",
            beta, fd.time, mean(sig), maximum(sig), minimum(sig), basename(f))
end

Colorbar(fig[1:2, 4], hm; label = "log₁₀ Σ_g / Σ₀", height = Relative(0.82),
         labelsize = 18, width = 18)
Label(fig[3, 1:3],
      "Σ₀ = 2ρ₀H is the initial surface density.  β = 3 was morphed from the β = 10 state at Ωt = 300; " *
      "the other runs started from the cold Q₀ = 1 initial condition.",
      fontsize = 12.5, color = :gray30, padding = (0, 0, 4, 4))
colgap!(fig.layout, 10); rowgap!(fig.layout, 8)

out = joinpath(HERE, "sc14_beta_colden.png")
save(out, fig; px_per_unit = px)
println("wrote $out")
