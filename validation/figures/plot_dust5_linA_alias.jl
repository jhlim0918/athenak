# Phase 5: why the linA mode lost half its growth rate on the refined mesh -- lattice
# aliasing at the outer level boundary.  Panels (a)-(c) are computed, not sketched: a 2D
# TSC deposit with the finest-level kernel (1D weights 1/8, 3/4, 1/8 at a cell centre) of
# the particle lattice drawn on top, shown on the cells that exist (fine cells left of the
# boundary, root cells = the 2x2 average of the fine image right of it).
#   (a) the generator's lattice, one particle per BLOCK cell, at t = 0: the interface
#       columns hold 7/8 and 1 + 1/16, everything else 1 (the dump at t = 0 agrees).
#   (b) the same lattice after an inward drift of 4.5 fine cells: the root particles now
#       sit at fine-cell centres inside the fine region and the fine kernel resolves the
#       lattice as a comb, 3/2 // 1/2 (the dump at t = 1.5: 4.49 / 1.51 about 3).
#   (c) problem/lattice_finest (lattice at the finest cell size everywhere, four
#       quarter-mass particles per root cell) after the same drift: exactly 1.
#   (d) the dumps: z-averaged dust density per column around the outer boundary of the
#       linA run with one particle per block cell, and the 3/2 // 1/2 prediction.
#
# Data: validation/run/dust5/linA/dust_column_means_outer_interface.txt (README there)
# Usage: julia validation/figures/plot_dust5_linA_alias.jl   [DUST5_RUN=<run/dust5 dir>]
using CairoMakie, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))
const HERE = @__DIR__
const RUN  = get(ENV, "DUST5_RUN", joinpath(dirname(HERE), "run", "dust5"))

# ---- the finest-level TSC deposit on a fine image; x in fine-cell units, boundary at 0
const NZ = 8                      # z rows of the strip (periodic; lattice period 2)
const XLO, XHI = -12, 12          # fine image extent (margin beyond the window)
function tsc(d)                   # d = distance of the particle from the cell centre
    a = abs(d)
    a < 0.5 ? 0.75 - a^2 : (a < 1.5 ? 0.5*(1.5 - a)^2 : 0.0)
end
function deposit(parts)           # parts: (x, z, mass); returns fine image [ix, iz]
    img = zeros(XHI - XLO, NZ)
    for (x, z, m) in parts
        i0 = floor(Int, x - XLO); j0 = floor(Int, z)
        for di in -1:1, dj in -1:1
            i = i0 + di; j = mod(j0 + dj, NZ)
            (1 <= i + 1 <= size(img, 1)) || continue
            img[i + 1, j + 1] += m * tsc(x - (XLO + i + 0.5)) * tsc(z - (j0 + dj + 0.5))
        end
    end
    img
end
# what the mesh holds: fine cells for x < 0, the 2x2 average for x >= 0 (shown on the fine image)
function onmesh(img)
    out = copy(img)
    for i in 1:2:size(img, 1), j in 1:2:NZ
        (XLO + i - 1 >= 0) || continue
        out[i:i+1, j:j+1] .= sum(img[i:i+1, j:j+1]) / 4
    end
    out
end
# lattices: fine = one particle (mass 1) per fine cell; block = one per BLOCK cell
fine_lat(x0, x1, m=1.0) = [(x + 0.5, z + 0.5, m) for x in x0:x1-1, z in 0:NZ-1]
root_lat(x0, x1)        = [(x + 1.0, z + 1.0, 4.0) for x in x0:2:x1-1, z in 0:2:NZ-1]
shift(parts, dx)        = [(x + dx, z, m) for (x, z, m) in parts]
block_lattice  = vcat(vec(fine_lat(XLO, 0)), vec(root_lat(0, XHI)))
finest_lattice = vcat(vec(fine_lat(XLO, 0)), vec(fine_lat(0, XHI, 1.0)))   # same mass per area
DRIFT = -4.5

fig = Figure(size=(1500, 900))
Label(fig[0, 1:2], "linA on the refined mesh: the outer level boundary (x = 3L/4), fine cells left, root cells right; the dust drifts inward (−x)",
      fontsize=16, font=:bold, tellwidth=false)

function panel!(pos, parts, title; xw=(-8, 6), annotate=true)
    ax = Axis(fig[pos...], title=title, xlabel="x  (fine cells from the level boundary)",
              ylabel="z  (fine cells)", aspect=DataAspect(), xticks=xw[1]:2:xw[2])
    img = onmesh(deposit(parts))
    xs = collect(XLO:XHI) .+ 0.0; zs = collect(0:NZ) .+ 0.0
    hm = heatmap!(ax, xs, zs, img, colormap=Reverse(:RdBu), colorrange=(0, 2))
    # cell edges: fine grid left of the boundary, root grid right of it
    vlines!(ax, collect(xw[1]:0), color=(:black, 0.35), linewidth=0.7)
    vlines!(ax, collect(0:2:xw[2]), color=(:black, 0.35), linewidth=0.7)

    for z in zs; lines!(ax, [xw[1], 0], [z, z], color=(:black, 0.35), linewidth=0.7); end
    for z in zs[1:2:end]; lines!(ax, [0, xw[2]], [z, z], color=(:black, 0.35), linewidth=0.7); end
    vlines!(ax, [0.0], color=:black, linewidth=3)
    # particles
    for (x, z, m) in parts
        (xw[1] <= x <= xw[2]) || continue
        scatter!(ax, [x], [z], color=(m > 2 ? :darkorange : :teal), markersize=(m > 2 ? 11 : 6),
                 strokecolor=:black, strokewidth=0.5)
    end
    # deposited value per column (the strip is uniform in z after the drift; label the mean)
    if annotate
        ix = xw[1]
        while ix < xw[2]
            w = ix < 0 ? 1 : 2
            i = ix - XLO + 1
            v = sum(img[i:i+w-1, :]) / (w * NZ)
            r = rationalize(v, tol=1e-9)
            s = denominator(r) == 1 ? string(numerator(r)) : "$(numerator(r))/$(denominator(r))"
            text!(ax, ix + w/2, NZ + 0.55, text=s, align=(:center, :bottom), fontsize=11,
                  color=(abs(v - 1) > 1e-9 ? :firebrick : :gray30))
            ix += w
        end
        text!(ax, xw[1], NZ + 1.7, text="deposited dust density / uniform, per column:",
              align=(:left, :bottom), fontsize=11, color=:gray30)
    end
    xlims!(ax, xw...); ylims!(ax, 0, NZ + 3)
    ax, hm
end

ax1, hm = panel!((1, 1), block_lattice, "(a) one particle per block cell, t = 0")
arrows2d!(ax1, [Point2f(4.0, NZ + 0.15)], [Vec2f(-2.0, 0.0)], color=:black, shaftwidth=2)
text!(ax1, 4.5, NZ - 0.1, text="drift", align=(:left, :top), fontsize=11)
panel!((1, 2), shift(block_lattice, DRIFT), "(b) the same lattice after a drift of 4.5 fine cells: root particles inside the fine region")
panel!((2, 1), shift(finest_lattice, DRIFT), "(c) lattice_finest: lattice at the finest cell size everywhere, after the same drift")
Colorbar(fig[1:2, 3], hm, label="deposited dust density / uniform", width=14)

# ---- (d) the dumps
ax4 = Axis(fig[2, 2], title="(d) the run: z-averaged dust density per column, one particle per block cell",
           xlabel="x  (fine cells from the level boundary; fine columns left)", ylabel="⟨ρ_p⟩_z / (ε ρ_0)")
f = joinpath(RUN, "linA", "dust_column_means_outer_interface.txt")
xb = 0.007854                     # the outer level boundary, 3L/4
if isfile(f)
    rows = Tuple{Float64,Vector{Float64},Vector{Float64}}[]
    for l in eachline(f)
        startswith(l, "t=") || continue
        tk = split(l); t = parse(Float64, tk[1][3:end])
        xv = Float64[]; vv = Float64[]
        for p in tk[2:end]
            a, b = split(p, ":"); push!(xv, parse(Float64, a)); push!(vv, parse(Float64, b))
        end
        push!(rows, (t, xv, vv))
    end
    # fine cell size from the first two (fine) columns
    h = rows[1][2][2] - rows[1][2][1]
    cols = Makie.wong_colors()
    for (k, t) in enumerate((0.0, 1.0, 1.5, 2.5, 3.5))
        r = rows[findfirst(r -> abs(r[1] - t) < 1e-6, rows)]
        scatterlines!(ax4, (r[2] .- xb) ./ h, r[3] ./ 3, color=cols[k], markersize=9,
                      label=@sprintf("t = %.1f", t))
    end
    hlines!(ax4, [1.5, 0.5], color=:firebrick, linestyle=:dash, label="3/2 // 1/2 (root particles at fine-cell centres)")
    hlines!(ax4, [1.0], color=:gray50, linestyle=:dot)
    vlines!(ax4, [0.0], color=:black, linewidth=2)
    axislegend(ax4, position=:rt, framevisible=false, nbanks=2, labelsize=11)
    xlims!(ax4, -10.5, 10.5); ylims!(ax4, 0.3, 1.9)
end

save(joinpath(HERE, "dust5_linA_alias.png"), fig, px_per_unit=1.5)
println("wrote ", joinpath(HERE, "dust5_linA_alias.png"))
