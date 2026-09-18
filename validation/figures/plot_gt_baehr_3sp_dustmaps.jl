# BR_3sp at t = 165, all three dust species summed.
# Figure 1: the gas surface density, the total dust surface density on the same grid, and
#   their ratio -- where the disc is metal-rich.
# Figure 2: the azimuthally (y-) averaged density in the x-z plane, gas and dust, with the
#   dust layer's thickness against the gas scale height.
# Input: validation/run/gtb_BR_3sp/dustmaps.npz (written by that directory's dustmaps.py).
# Usage: julia validation/figures/plot_gt_baehr_3sp_dustmaps.jl
using CairoMakie, NPZ, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
d = npzread(joinpath(dirname(HERE), "run", "gtb_BR_3sp", "dustmaps.npz"))
Hg = 2.16878
sg = d["sig_g"]; sd = d["sig_d"]; t = d["t"]
x = range(d["x1min"]/Hg, d["x1max"]/Hg, length=size(sg, 2))
y = range(d["x2min"]/Hg, d["x2max"]/Hg, length=size(sg, 1))
Zbar = sum(sd)/sum(sg)

# ---------------- figure 1: surface densities ----------------
fig = Figure(size=(1450, 470))
ax = Axis(fig[1,1], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
          title=@sprintf("gas  Σ_g          max/mean = %.2f", maximum(sg)/mean(sg)))
hm = heatmap!(ax, x, y, permutedims(sg), colormap=:cividis)
Colorbar(fig[1,2], hm, label="Σ_g")

ax = Axis(fig[1,3], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
          title=@sprintf("dust  Σ_d  (all 3 sizes)    max/mean = %.0f", maximum(sd)/mean(sd)))
hm = heatmap!(ax, x, y, permutedims(sd), colormap=:cividis, colorscale=log10,
              colorrange=(mean(sd)/10, maximum(sd)))
Colorbar(fig[1,4], hm, label="Σ_d")

ax = Axis(fig[1,5], xlabel="x / H_g", ylabel="y / H_g", aspect=DataAspect(),
          title=@sprintf("Σ_d / Σ_g        max = %.2f  (%.0f× Z̄)",
                         maximum(sd./sg), maximum(sd./sg)/Zbar))
hm = heatmap!(ax, x, y, permutedims(sd./sg), colormap=:cividis, colorscale=log10,
              colorrange=(Zbar/5, maximum(sd./sg)))
Colorbar(fig[1,6], hm, label="Σ_d/Σ_g")
Label(fig[0,:], @sprintf("BR_3sp, t = %.0f Ω⁻¹: total dust (St = 0.01, 0.1, 1), Z̄ = %.3f",
      t, Zbar), fontsize=17, font=:bold)
save(joinpath(HERE, "gt_baehr_3sp_sigma_total.png"), fig, px_per_unit=2)

# ---------------- figure 2: the x-z plane ----------------
rg = d["rhog_xz"]; rd = d["rhod_xz"]
z = range(d["x3min"]/Hg, d["x3max"]/Hg, length=size(rg, 1))
zc = collect(z)
fig2 = Figure(size=(1400, 560))

ax = Axis(fig2[1,1], xlabel="x / H_g", ylabel="z / H_g",
          title="gas  ⟨ρ_g⟩_y")
hm = heatmap!(ax, x, z, permutedims(rg), colormap=:cividis, colorscale=log10,
              colorrange=(maximum(rg)/1e3, maximum(rg)))
Colorbar(fig2[1,2], hm, label="⟨ρ_g⟩_y")
ylims!(ax, -3, 3)

ax = Axis(fig2[1,3], xlabel="x / H_g", ylabel="z / H_g",
          title=@sprintf("dust  ⟨ρ_d⟩_y   (all 3 sizes)   max %.3f", maximum(rd)))
hm = heatmap!(ax, x, z, permutedims(rd), colormap=:cividis, colorscale=log10,
              colorrange=(maximum(rd)/1e3, maximum(rd)))
Colorbar(fig2[1,4], hm, label="⟨ρ_d⟩_y")
ylims!(ax, -3, 3)

# vertical profiles, x-averaged
ax = Axis(fig2[1,5], xlabel="⟨ρ⟩_xy  (normalized)", ylabel="z / H_g", xscale=log10,
          title="vertical structure")
pg = vec(mean(rg, dims=2)); pd = vec(mean(rd, dims=2))
lines!(ax, pg./maximum(pg), zc, color=:black, linewidth=2, label="gas")
lines!(ax, pd./maximum(pd), zc, color=:crimson, linewidth=2, label="dust (total)")
ylims!(ax, -3, 3); xlims!(ax, 1e-4, 2)
axislegend(ax, position=:rb, framevisible=false, labelsize=12)
colsize!(fig2.layout, 5, Relative(0.17))
Label(fig2[0,:], @sprintf("BR_3sp, t = %.0f Ω⁻¹: azimuthally averaged density, x–z plane", t),
      fontsize=17, font=:bold)
save(joinpath(HERE, "gt_baehr_3sp_rho_xz.png"), fig2, px_per_unit=2)

# the dust layer thickness of the x-averaged profile
w(p) = sqrt(sum(p .* zc.^2)/sum(p))
@printf("mass-weighted sqrt<z^2>:  gas %.3f H_g   dust %.3f H_g   ratio %.3f\n",
        w(pg), w(pd), w(pd)/w(pg))
@printf("Sigma_d/Sigma_g: mean %.4f  max %.3f\n", Zbar, maximum(sd./sg))
println("wrote gt_baehr_3sp_sigma_total.png and gt_baehr_3sp_rho_xz.png")
