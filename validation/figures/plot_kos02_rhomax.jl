# Replicas of Kim, Ostriker & Stone (2002) Figs. 4 and 13: maximum density against time for
# the 3-D models of their Table 1, plotted as log10(rho_max/rho00 - 1) against orbits like
# theirs. (a) A, B (Q = 0.7, s0 = 25, no self-gravity), (b) C, D, E (with self-gravity),
# (c) F, G (Q = 1.5, s0 = 1), (d) H, I, J (Q = 1, s0 = 1). A cross marks a run stopped by
# its wall-time cap before the KOS02 end time.
# Data: validation/run/kos02/tier2/reduced/<model>.npz from tier2_reduce.py; models.json.
# Run from validation/: julia figures/plot_kos02_rhomax.jl
using CairoMakie, NPZ
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))

const HERE = @__DIR__
const DIR = joinpath(dirname(HERE), "run", "kos02", "tier2")

# models.json, read without a JSON package: one {...} object per model
models = Dict{String,Dict{String,Any}}()
for blk in eachmatch(r"\{[^{}]*\}", read(joinpath(DIR, "models.json"), String))
    d = Dict{String,Any}()
    for kv in eachmatch(r"\"(\w+)\":\s*(\"[^\"]*\"|[-0-9.eE+]+|true|false)", blk.match)
        v = kv[2]
        d[kv[1]] = startswith(v, "\"") ? strip(v, '"') : v in ("true", "false") ? v == "true" : parse(Float64, v)
    end
    models[d["name"]] = d
end
function curve(m)
    d = npzread(joinpath(DIR, "reduced", m * ".npz")); r = models[m]
    t = d["t"] ./ r["torb"]; y = log10.(max.(d["rhomax"] ./ r["rho00"] .- 1, 1e-3))
    (t, y, t[end] < r["norb"] - 1e-3)
end

fig = Figure(size=(1000, 820))
panels = [
    (fig[1, 1], "(a) Q = 0.7, s₀ = 25, without self-gravity", [("A", "A: q = 0, β = 1", :solid), ("B", "B: q = 1, β = 1", :dot)], 4),
    (fig[1, 2], "(b) Q = 0.7, s₀ = 25, with self-gravity", [("D", "D: q = 1, β = 1", :solid), ("C", "C: q = 1, β = ∞", :dot), ("E", "E: q = 0, β = 1", :dash)], 4),
    (fig[2, 1], "(c) Q = 1.5, s₀ = 1, β = 1", [("F", "F: G = 0", :dot), ("G", "G: G ≠ 0", :solid)], 5),
    (fig[2, 2], "(d) Q = 1.0, s₀ = 1", [("H", "H: G = 0, β = 1", :dot), ("I", "I: G ≠ 0, β = 1", :solid), ("J", "J: G ≠ 0, β = ∞", :dash)], 5)]
cols = Dict("A" => "#2a78d6", "B" => "#eb6834", "C" => "#eb6834", "D" => "#2a78d6", "E" => "#2e9e5b",
            "F" => "#eb6834", "G" => "#2a78d6", "H" => "#eb6834", "I" => "#2a78d6", "J" => "#2e9e5b")
for (gp, ttl, runs, tmax) in panels
    ax = Axis(gp, title=ttl, xlabel="orbits", ylabel=rich("log₁₀ (ρ", subscript("max"), " / ρ", subscript("00"), " − 1)"),
              titlesize=13, titlealign=:left)
    for (m, lab, ls) in runs
        t, y, capped = curve(m)
        lines!(ax, t, y, color=cols[m], linestyle=ls, linewidth=2, label=lab)
        capped && scatter!(ax, [t[end]], [min(y[end], 0.95)], color=cols[m], marker=:xcross, markersize=11)
    end
    xlims!(ax, 0, tmax); ylims!(ax, -2.1, 1)
    axislegend(ax, position=:lt, framevisible=false, labelsize=11)
end
Label(fig[0, 1:2], "AthenaK runs of KOS02 Table 1: maximum density (cf. KOS02 Figs. 4 and 13)", fontsize=16, font=:bold)
Label(fig[3, 1:2], "× : run stopped by its wall-time cap (E and I in collapse or near it; A and H by a small Alfvén time step in the corona).",
      fontsize=11, color=:gray30)
out = joinpath(HERE, "kos02_rhomax.png")
save(out, fig, px_per_unit=2)
println("wrote ", out)
