# Error-map twins of the refined-region figures (Figs. 4 and 6): the midplane density
# perturbation of each run minus that of a uniform run, (rho_run - rho_ref)/(rho0 A), with
# the MeshBlock outlines. All maps are first averaged onto the 128^2 root grid (2x2 means
# of the level-1 cells, which conserve mass), so the difference is taken cell by cell at one
# resolution and sub-cell structure of the finer maps does not show up as error.
#   fargo_amr_errmap_hydro.png -- hydro swing at t = 8: ring and patch minus uniform 128^2
#       (top); uniform 128^2, ring and patch minus uniform 256^2 (bottom)
#   fargo_amr_errmap_mhd.png   -- MHD swing (B0 = 1): static ring and static patch at t = 6.3,
#       moving ring at t = k pi/4, minus the uniform 128^2 twin (mesh_smr.athinput with
#       refinement off) run at CFL 0.15: the refined runs step at the level-1 time step, and
#       the CFL 0.3 and 0.15 uniform twins differ by rms 0.004-0.013 at equal times, as much as
#       the refined-minus-uniform differences, so the twin must take the same step
# Printed: mean |error| in the band within 2 root cells of a level change, in the rest of
# the level-1 region and in the rest of the level-0 region.
# Slices: validation/run/fargo_amr/meshfig/*.{f32,txt} from slice.py (README).
# Run from validation/: julia figures/plot_fargo_amr_errmaps.jl
using CairoMakie
using Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const HERE = @__DIR__
const DIR = joinpath(dirname(HERE), "run", "fargo_amr", "meshfig")
const A = 1e-4
const NR = 128

function readslice(name)
    t = joinpath(DIR, name * ".txt")
    hdr = split(strip(first(eachline(t)), ['#', ' ']))
    time = parse(Float64, hdr[1]); N = parse(Int, hdr[2]); ext = parse.(Float64, hdr[3:6])
    img = Array{Float32}(undef, N, N)
    read!(joinpath(DIR, name * ".f32"), img)
    rows = [parse.(Float64, split(l)) for l in eachline(t) if !startswith(l, "#") && !isempty(strip(l))]
    (; time, N, ext, img, rows)
end
# perturbation (rho - rho0)/(rho0 A) averaged onto the root grid; in Julia img is [i, j] (x, y)
# (slice.py writes C order [j][i], read here column-major)
function root(s)
    r = s.N ÷ NR
    p = (Float64.(s.img) .- 1.0) ./ A
    r == 1 && return p
    [sum(@view p[(i-1)*r+1:i*r, (j-1)*r+1:j*r]) / r^2 for i in 1:NR, j in 1:NR]
end
# level-1 mask on the root grid from the block rows (x0 x1 y0 y1 pixels-per-cell)
function finemask(s)
    m = falses(NR, NR)
    s.N == NR && return m
    dx = (s.ext[2] - s.ext[1]) / NR
    for r in s.rows
        r[5] == 1 || continue          # one pixel per cell: a level-1 block
        i0 = round(Int, (r[1] - s.ext[1]) / dx) + 1; i1 = round(Int, (r[2] - s.ext[1]) / dx)
        j0 = round(Int, (r[3] - s.ext[3]) / dx) + 1; j1 = round(Int, (r[4] - s.ext[3]) / dx)
        m[i0:i1, j0:j1] .= true
    end
    m
end
# cells within w root cells of a level change (periodic in x and y)
function band(m; w=2)
    [any(m[mod1(i + di, NR), mod1(j + dj, NR)] != m[i, j] for di in -w:w, dj in -w:w)
     for i in 1:NR, j in 1:NR]
end
mean_abs(e, sel) = any(sel) ? sum(abs, e[sel]) / count(sel) : NaN
function stats(lab, e, m)
    b = band(m)
    @printf("%-34s rms %.4f  max %.4f | mean|e|: band %.4f, rest of level 1 %.4f, rest of level 0 %.4f\n",
            lab, sqrt(sum(abs2, e) / length(e)), maximum(abs, e), mean_abs(e, b),
            mean_abs(e, m .& .!b), mean_abs(e, .!m .& .!b))
end
function panel!(gp, s, e, title; cr, ylab=true)
    xs = range(s.ext[1], s.ext[2], length=NR + 1); ys = range(s.ext[3], s.ext[4], length=NR + 1)
    ax = Axis(gp, aspect=DataAspect(), title=title, xlabel="x / H", ylabel=(ylab ? "y / H" : ""),
              xgridvisible=false, ygridvisible=false, titlesize=12)
    hm = heatmap!(ax, xs, ys, e, colormap=:balance, colorrange=cr)
    rmax = maximum(r[5] for r in s.rows)
    for r in s.rows
        L = round(Int, log2(rmax / r[5]))
        x0, x1, y0, y1 = r[1:4]
        lines!(ax, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
               color=(L == 0 ? (:gray30, 0.6) : ("#7b3294", 0.8)), linewidth=(L == 0 ? 0.6 : 0.5))
    end
    limits!(ax, s.ext...)
    hm
end

# --- hydro (Fig. 4 twin)
u, u2 = readslice("hydro_unif"), readslice("hydro_unif256")
ru, ru2 = root(u), root(u2)
ri, pa = readslice("hydro_ring"), readslice("hydro_patch")
println("hydro, t = 8:")
fig = Figure(size=(1420, 900))
cr = (-0.15, 0.15)
local hm
for (n, (s, lab)) in enumerate([(ri, "ring"), (pa, "patch")])
    e = root(s) .- ru
    stats("$lab − uniform 128²", e, finemask(s))
    global hm = panel!(fig[1, n+1], s, e, "$lab − uniform 128²"; cr=cr, ylab=false)
end
for (n, (s, lab)) in enumerate([(u, "uniform 128²"), (ri, "ring"), (pa, "patch")])
    e = root(s) .- ru2
    stats("$lab − uniform 256²", e, finemask(s))
    panel!(fig[2, n], s, e, "$lab − uniform 256²"; cr=cr, ylab=(n == 1))
end
Label(fig[1, 1], "hydro swing, t = 8\n\ntop: refined run minus\nuniform 128²\nbottom: minus uniform 256²\n(all averaged to the\n128² root grid)\n\nwave amplitude\n|ρ − ρ₀|/(ρ₀A) ≈ 3.6",
      fontsize=12, color=:gray20, tellheight=false, tellwidth=false)
for c in 1:3; colsize!(fig.layout, c, Fixed(360)); end
Colorbar(fig[1:2, 4], hm, label="Δ(ρ − ρ₀) / (ρ₀ A)")
Label(fig[3, 1:3], "MeshBlock outlines: gray level 0 (root 128², 16³ blocks), purple level 1",
      fontsize=11, color=:gray30)
save(joinpath(HERE, "fargo_amr_errmap_hydro.png"), fig, px_per_unit=2)
println("wrote figures/fargo_amr_errmap_hydro.png")

# --- MHD (Fig. 6 twin)
isfile(joinpath(DIR, "mhd_uni15_6.txt")) || (println("MHD twin slices missing; skipped"); exit())
println("MHD, B0 = 1:")
panels = [("mhd_smr_last", "mhd_uni15_last", "static ring"), ("mhd_ypatch_last", "mhd_uni15_last", "static patch"),
          ("mhd_amr_1", "mhd_uni15_1", "moving ring"), ("mhd_amr_2", "mhd_uni15_2", "moving ring"),
          ("mhd_amr_4", "mhd_uni15_4", "moving ring"), ("mhd_amr_6", "mhd_uni15_6", "moving ring")]
fig = Figure(size=(1250, 860))
cr = (-0.03, 0.03)
for (n, (f, tw, lab)) in enumerate(panels)
    s, w = readslice(f), readslice(tw)
    e = root(s) .- root(w)
    stats(@sprintf("%s t=%.3f (twin t=%.3f)", lab, s.time, w.time), e, finemask(s))
    i, j = divrem(n - 1, 3)
    global hm = panel!(fig[i+1, j+1], s, e, @sprintf("%s − uniform 128², t = %.2f", lab, s.time); cr=cr, ylab=(j == 0))
end
Colorbar(fig[1:2, 4], hm, label="Δ(ρ − ρ₀) / (ρ₀ A)")
Label(fig[3, 1:3], "B₀ = 1; uniform twin at CFL 0.15 (the refined runs' step); both maps averaged to the 128² root grid; wave amplitude 0.2–0.9. Outlines: gray level 0, purple level 1;\n" *
      "moving ring refined where blocks overlap x_c(t) ± π/16, x_c = (π/4) sin t",
      fontsize=11, color=:gray30)
save(joinpath(HERE, "fargo_amr_errmap_mhd.png"), fig, px_per_unit=2)
println("wrote figures/fargo_amr_errmap_mhd.png")
