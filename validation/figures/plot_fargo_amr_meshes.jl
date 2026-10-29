# Where the refined runs of Figs. 3 and 4 are refined: midplane density perturbation
# (rho - rho0)/(rho0 A) with the MeshBlock outlines (gray level 0, red level 1).
#   fargo_amr_mesh_hydro.png -- the hydro swing with gravity (Fig. 3) at t = 8: uniform,
#                               ring (6 of 8 block columns), patch (the ring cut to the
#                               middle 4 of 8 block rows)
#   fargo_amr_mesh_mhd.png   -- the MHD swing with gravity (Fig. 4, B0 = 1): static ring and
#                               static patch at t = 6.3, and the moving ring (prescribed
#                               window x_c = (pi/4) sin t, half-width pi/16) at four times
# Slices: validation/run/fargo_amr/meshfig/*.{f32,txt} from slice.py (see README).
# Run from validation/: julia figures/plot_fargo_amr_meshes.jl
using CairoMakie
using Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const HERE = @__DIR__
const DIR = joinpath(dirname(HERE), "run", "fargo_amr", "meshfig")
const A = 1e-4

function readslice(name)
    t = joinpath(DIR, name * ".txt")
    hdr = split(strip(first(eachline(t)), ['#', ' ']))
    time = parse(Float64, hdr[1]); N = parse(Int, hdr[2]); ext = parse.(Float64, hdr[3:6])
    img = Array{Float32}(undef, N, N)
    read!(joinpath(DIR, name * ".f32"), img)
    rows = [parse.(Float64, split(l)) for l in eachline(t) if !startswith(l, "#") && !isempty(strip(l))]
    (; time, N, ext, img, rows)
end
function panel!(gp, s, title; cr, ylab=true)
    xs = range(s.ext[1], s.ext[2], length=s.N + 1); ys = range(s.ext[3], s.ext[4], length=s.N + 1)
    ax = Axis(gp, aspect=DataAspect(), title=title, xlabel="x / H",
              ylabel=(ylab ? "y / H" : ""), xgridvisible=false, ygridvisible=false)
    hm = heatmap!(ax, xs, ys, (s.img .- 1f0) ./ Float32(A), colormap=:cividis, colorrange=cr)
    rmax = maximum(r[5] for r in s.rows)
    for r in s.rows
        L = round(Int, log2(rmax / r[5]))
        x0, x1, y0, y1 = r[1:4]
        lines!(ax, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
               color=(L == 0 ? (:gray20, 0.8) : ("#d62728", 0.9)), linewidth=(L == 0 ? 0.8 : 0.5))
    end
    limits!(ax, s.ext...)
    hm
end

# --- hydro (Fig. 3)
fig = Figure(size=(1250, 470))
hms = Any[]
for (n, (f, lab)) in enumerate([("hydro_unif", "uniform 128²"), ("hydro_ring", "ring"),
                                ("hydro_patch", "patch")])
    s = readslice(f)
    push!(hms, panel!(fig[1, n], s, @sprintf("%s, t = %.1f", lab, s.time); cr=(-3, 3), ylab=(n == 1)))
end
Colorbar(fig[1, 4], hms[1], label="(ρ − ρ₀) / (ρ₀ A)")
Label(fig[2, 1:3], "MeshBlock outlines: dark gray level 0 (root 128², 16³ blocks), red level 1",
      fontsize=11, color=:gray30)
save(joinpath(HERE, "fargo_amr_mesh_hydro.png"), fig)
println("wrote figures/fargo_amr_mesh_hydro.png")

# --- MHD (Fig. 4)
isfile(joinpath(DIR, "mhd_amr_6.txt")) || (println("MHD slices missing; skipped"); exit())
fig = Figure(size=(1250, 860))
panels = [("mhd_smr_last", "static ring"), ("mhd_ypatch_last", "static patch"),
          ("mhd_amr_1", "moving ring"), ("mhd_amr_2", "moving ring"),
          ("mhd_amr_4", "moving ring"), ("mhd_amr_6", "moving ring")]
ss = [readslice(f) for (f, _) in panels]
c = maximum(maximum(abs.((s.img .- 1f0) ./ Float32(A))) for s in ss)
c = ceil(c)
for (n, ((f, lab), s)) in enumerate(zip(panels, ss))
    i, j = divrem(n - 1, 3)
    push!(hms, panel!(fig[i+1, j+1], s, @sprintf("%s, t = %.2f", lab, s.time); cr=(-c, c), ylab=(j == 0)))
end
Colorbar(fig[1:2, 4], hms[end], label="(ρ − ρ₀) / (ρ₀ A)")
Label(fig[3, 1:3], "B₀ = 1. MeshBlock outlines: dark gray level 0 (root 128², 16³ blocks), red level 1; " *
      "moving ring: refined where blocks overlap x_c(t) ± π/16, x_c = (π/4) sin t",
      fontsize=11, color=:gray30)
save(joinpath(HERE, "fargo_amr_mesh_mhd.png"), fig)
println("wrote figures/fargo_amr_mesh_mhd.png")
