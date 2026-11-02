# Surface density maps of the self-gravitating Q = 0.7, s0 = 25 models at the times of
# Kim, Ostriker & Stone (2002) Fig. 10 (model C: 2.0, 2.9, 3.4 orbits) and Fig. 11 (model D:
# 2.0, 3.3, 4.1 orbits; our run ends at 4.0), and model E (their Fig. 14 is at 0.5 orbit;
# runaway at 0.75), as log10(Sigma/Sigma0).
# Data: validation/run/kos02/tier2/reduced/<model>.npz from tier2_reduce.py.
# Run from validation/: julia figures/plot_kos02_sigma.jl
using CairoMakie, NPZ, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const HERE = @__DIR__
const DIR = joinpath(dirname(HERE), "run", "kos02", "tier2")
const SIG0 = 4.0
torb(m) = parse(Float64, match(Regex("\"name\": \"$m\"[^}]*\"torb\": ([0-9.]+)"), read(joinpath(DIR, "models.json"), String))[1])
Hof(m) = parse(Float64, match(Regex("\"name\": \"$m\"[^}]*\"H\": ([0-9.]+)"), read(joinpath(DIR, "models.json"), String))[1])

rows = [("C", "C (β = ∞)", [2.0, 2.9, 3.4]), ("D", "D (β = 1)", [2.0, 3.3, 4.0]), ("E", "E (q = 0, β = 1)", [0.5, 0.75, 1.5])]
fig = Figure(size=(1050, 1060))
hm = nothing
for (r, (m, lab, ts)) in enumerate(rows)
    d = npzread(joinpath(DIR, "reduced", m * ".npz"))
    t = d["t"] ./ torb(m); ext = d["ext"] ./ Hof(m); S = d["sigma"]
    for (c, tt) in enumerate(ts)
        i = argmin(abs.(t .- tt))
        img = log10.(max.(permutedims(S[i, :, :]) ./ SIG0, 1e-2))     # [x, y]
        ax = Axis(fig[r, c], aspect=DataAspect(), title=@sprintf("%s, t = %.2f orbits", lab, t[i]),
                  xlabel=r == 3 ? "x / H" : "", ylabel=c == 1 ? "y / H" : "", titlesize=13)
        global hm = heatmap!(ax, range(ext[1], ext[2], length=size(img, 1) + 1), range(ext[3], ext[4], length=size(img, 2) + 1),
                      img, colormap=:cividis, colorrange=(-0.5, 1.5))
        text!(ax, ext[1] + 0.4, ext[4] - 0.4, text=@sprintf("max %.3g Σ₀", maximum(S[i, :, :]) / SIG0),
              align=(:left, :top), color=:black, fontsize=11)
        c > 1 && hideydecorations!(ax, grid=false, ticks=false)
        r < 3 && hidexdecorations!(ax, grid=false, ticks=false)
    end
end
Colorbar(fig[1:3, 4], hm, label="log₁₀ Σ / Σ₀ (clipped at 1.5)")
Label(fig[0, 1:3], "Surface density, Q = 0.7, s₀ = 25 (cf. KOS02 Figs. 10, 11, 14)", fontsize=16, font=:bold)
out = joinpath(HERE, "kos02_sigma.png")
save(out, fig, px_per_unit=2)
println("wrote ", out)
