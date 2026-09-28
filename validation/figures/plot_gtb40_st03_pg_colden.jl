# gtb40_st03_pg: Baehr+22 box at 40 cells per H_g, one dust species St = 0.3, Z = 0.02,
# radial pressure gradient eta v_K = 0.05 cs.  Gas and dust surface density on the same
# grid at the same instant, their ratio, and a zoom on the densest dust column.
# Input: validation/run/gtb40_st03_pg/colden.npz (from that directory's colden.py).
# Usage: julia validation/figures/plot_gtb40_st03_pg_colden.jl
using CairoMakie, NPZ, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
d = npzread(joinpath(dirname(HERE), "run", "gtb40_st03_pg", "colden.npz"))
Hg = 2.16878
sg = Float64.(d["sig_g"]); sd = Float64.(d["sig_d"]); t = d["t_gas"]
nx, ny = size(sg, 2), size(sg, 1)
x = range(d["x1min"]/Hg, d["x1max"]/Hg, length=nx+1); x = (x[1:end-1] .+ x[2:end]) ./ 2
y = range(d["x2min"]/Hg, d["x2max"]/Hg, length=ny+1); y = (y[1:end-1] .+ y[2:end]) ./ 2
Zbar = sum(sd)/sum(sg); zr = sd ./ sg
floor_d = mean(sd)/20
sdp = max.(sd, floor_d)

fig = Figure(size=(1300, 1150))
ax = Axis(fig[1,1], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
          title=@sprintf("gas  Σ_g     max/mean = %.2f", maximum(sg)/mean(sg)))
hm = heatmap!(ax, x, y, permutedims(sg), colormap=:cividis)
Colorbar(fig[1,2], hm, label="Σ_g")

ax = Axis(fig[1,3], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
          title=@sprintf("dust  Σ_d  (St = 0.3)     max/mean = %.0f", maximum(sd)/mean(sd)))
hm = heatmap!(ax, x, y, permutedims(sdp), colormap=:cividis, colorscale=log10,
              colorrange=(floor_d, maximum(sd)))
Colorbar(fig[1,4], hm, label="Σ_d")

ax = Axis(fig[2,1], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
          title=@sprintf("Σ_d / Σ_g     max = %.2f  (%.0f× Z̄ = %.3f)", maximum(zr), maximum(zr)/Zbar, Zbar))
zrp = max.(zr, Zbar/20)
hm = heatmap!(ax, x, y, permutedims(zrp), colormap=:cividis, colorscale=log10,
              colorrange=(Zbar/20, maximum(zr)))
Colorbar(fig[2,2], hm, label="Σ_d/Σ_g")

# zoom on the densest dust column, +-1.5 H_g
jm, im = Tuple(argmax(sd))
w = round(Int, 1.5*Hg/((d["x1max"]-d["x1min"])/nx))
ii = max(1, im-w):min(nx, im+w); jj = max(1, jm-w):min(ny, jm+w)
ax = Axis(fig[2,3], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
          title=@sprintf("zoom on the peak, (x, y) = (%.2f, %.2f) H_g", x[im], y[jm]))
hm = heatmap!(ax, x[ii], y[jj], permutedims(sdp[jj, ii]), colormap=:cividis, colorscale=log10,
              colorrange=(floor_d, maximum(sd)))
contour!(ax, x[ii], y[jj], permutedims(sg[jj, ii]), levels=6, color=(:white, 0.6), linewidth=1)
Colorbar(fig[2,4], hm, label="Σ_d   (white: Σ_g contours)")

Label(fig[0,:], @sprintf("gtb40_st03_pg,  t = %.1f Ω⁻¹:  Baehr box at 40 cells/H_g, St = 0.3, Z = %.3f, ηv_K = 0.05 c_s",
      t, Zbar), fontsize=17, font=:bold)
resize_to_layout!(fig)
save(joinpath(HERE, "gtb40_st03_pg_colden.png"), fig, px_per_unit=1.5)
println("wrote gtb40_st03_pg_colden.png")
@printf("peak Sigma_d %.3f at (x,y) = (%.3f, %.3f) H_g; Sigma_g there %.3f; local ratio %.2f\n",
        sd[jm,im], x[im], y[jm], sg[jm,im], zr[jm,im])
