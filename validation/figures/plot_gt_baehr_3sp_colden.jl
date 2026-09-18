# BR_3sp at t = 165: the gas surface density of the gravito-turbulent box and the surface
# density of each dust species on the same grid, plus the local dust-to-gas ratio of the
# St = 1 population.  The question the figure answers: which grain sizes trace the gas
# spirals and which collect inside them.
# Input: validation/run/gtb_BR_3sp/colden3sp.npz (written by that directory's colden3sp.py).
# Usage: julia validation/figures/plot_gt_baehr_3sp_colden.jl
using CairoMakie, NPZ, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
d = npzread(joinpath(dirname(HERE), "run", "gtb_BR_3sp", "colden3sp.npz"))
Hg = 2.16878
sg = d["sig_g"]; t = d["t"]
x = range(d["x1min"]/Hg, d["x1max"]/Hg, length=size(sg, 2))
y = range(d["x2min"]/Hg, d["x2max"]/Hg, length=size(sg, 1))
St = [0.01, 0.1, 1.0]
sd = [d["sig_d$(s)"] for s in 0:2]
Zbar = 0.04/3

fig = Figure(size=(1500, 820))

ax = Axis(fig[1,1], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
          title=@sprintf("gas  Σ_g        max/mean = %.2f", maximum(sg)/mean(sg)))
hm = heatmap!(ax, x, y, permutedims(sg), colormap=:cividis)
Colorbar(fig[1,2], hm, label="Σ_g")

for (i, s) in enumerate(1:3)
    r = i == 1 ? 1 : 2
    c = i == 1 ? 3 : (i == 2 ? 1 : 3)
    ax = Axis(fig[r, c], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
              title=@sprintf("dust St = %g      max/mean = %.0f", St[i], maximum(sd[i])/mean(sd[i])))
    hm = heatmap!(ax, x, y, permutedims(sd[i]), colormap=:cividis,
                  colorscale=log10, colorrange=(mean(sd[i])/8, maximum(sd[3])))
    Colorbar(fig[r, c+1], hm, label="Σ_d")
end

# local metallicity of the settled population
Zloc = sd[3] ./ sg
ax = Axis(fig[2,5], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
          title=@sprintf("St = 1:  Σ_d/Σ_g      max = %.2f  (%.0f× Z̄)",
                         maximum(Zloc), maximum(Zloc)/Zbar))
hm = heatmap!(ax, x, y, permutedims(Zloc), colormap=:cividis, colorscale=log10,
              colorrange=(Zbar/4, maximum(Zloc)))
Colorbar(fig[2,6], hm, label="Σ_d/Σ_g")
Label(fig[1,5:6], @sprintf("BR_3sp,  t = %.0f Ω⁻¹\nZ = 0.04, back-reaction on\n\nΣ_g mean %.2f\nΣ_d mean %.4f per species\n\nsmall grains trace the gas;\nSt = 1 collects in the spirals",
      t, mean(sg), mean(sd[1])), fontsize=13, tellwidth=false, tellheight=false)

save(joinpath(HERE, "gt_baehr_3sp_colden.png"), fig, px_per_unit=2)
println("wrote gt_baehr_3sp_colden.png")
for i in 1:3
    @printf("St=%-5g  Sigma_d max/mean = %6.1f   max local Z = %.3f (%.0fx mean)\n",
            St[i], maximum(sd[i])/mean(sd[i]), maximum(sd[i]./sg), maximum(sd[i]./sg)/Zbar)
end
