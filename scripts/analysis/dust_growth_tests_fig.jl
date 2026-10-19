# Figure of the dust growth tests D3 (scripts/analysis/dust_growth_tests.sh):
#   julia scripts/analysis/dust_growth_tests_fig.jl [validation/run/dust_growth]
# writes validation/figures/dust_growth_tests.png
#   (b) uniform growth vs the analytic S(t), and its error at three timesteps
#   (c) the turbulent fragmentation cap
#   (e) growth through AMR refine/derefine with particle splitting; dashed: restart
#       at t ~ 0.5 with a zero growth rate (constant through the split)
using CairoMakie

run = length(ARGS) > 0 ? ARGS[1] : "validation/run/dust_growth"

function read_phst(fn)
    cols = String[]
    rows = Vector{Vector{Float64}}()
    for l in eachline(fn)
        if startswith(l, "#")
            if isempty(cols) && occursin("[1]=", l)
                cols = [m.captures[1] for m in eachmatch(r"\[\d+\]=(\S+)", l)]
            end
            continue
        end
        push!(rows, parse.(Float64, split(l)))
    end
    d = reduce(hcat, rows)'
    Dict(c => d[:, i] for (i, c) in enumerate(cols))
end

coef = sqrt(pi / 8)
sexact(t; s0=1e-3, rhod=1.0, alpha=1e-3) = (sqrt(s0) + 0.5 * coef * rhod * sqrt(3alpha) * t)^2

fig = Figure(size=(1200, 380), fontsize=13)
cols = Makie.wong_colors()

ax1 = Axis(fig[1, 1], xlabel="t Ω", ylabel="S", yscale=log10,
           title="(b) uniform growth, no fragmentation")
ax1b = Axis(fig[1, 2], xlabel="t Ω", ylabel="|S/S_exact − 1|", yscale=log10,
            title="(b) error vs timestep")
tt = range(0, 20, length=200)
lines!(ax1, tt, sexact.(tt), color=:black, linewidth=1.5, label="analytic")
for (n, c) in enumerate(("0.45", "0.225", "0.1125"))
    h = read_phst(joinpath(run, "b", "cfl$c", "dgrow_box.phst"))
    if n == 1
        scatter!(ax1, h["time"][1:4:end], h["St_mean"][1:4:end], color=cols[1],
                 markersize=6, label="AthenaK (C = 0.45)")
    end
    err = abs.(h["St_mean"] ./ sexact.(h["time"]) .- 1)
    lines!(ax1b, h["time"][2:end], err[2:end], color=cols[n], label="C = $c")
end
axislegend(ax1, position=:rb, framevisible=false)
axislegend(ax1b, position=:rb, framevisible=false)

ax2 = Axis(fig[1, 3], xlabel="t Ω", ylabel="S", title="(c) fragmentation cap")
sfrag = 0.01^2 / (3e-3)
h = read_phst(joinpath(run, "c", "dgrow_box.phst"))
lines!(ax2, tt, min.(sexact.(tt), 1.0), color=:black, linewidth=1, linestyle=:dot,
       label="growth law")
hlines!(ax2, [sfrag], color=:gray, linestyle=:dash, label="S_frag = v_f²/(3α)")
lines!(ax2, h["time"], h["St_mean"], color=cols[2], linewidth=2, label="AthenaK")
ylims!(ax2, 0, 0.05)
axislegend(ax2, position=:rb, framevisible=false)

ax3 = Axis(fig[1, 4], xlabel="t Ω", ylabel="S", yscale=log10,
           title="(e) AMR + particle splitting, 4 ranks")
ha = read_phst(joinpath(run, "e", "amr", "nsh3d_amr.phst"))
hs = read_phst(joinpath(run, "e", "split", "nsh3d_amr.phst"))
for (n, (k, lab)) in enumerate((("St_max", "max"), ("St_mean", "mean"), ("St_min", "min")))
    lines!(ax3, ha["time"], ha[k], color=cols[n], label=lab)
    lines!(ax3, hs["time"], hs[k], color=cols[n], linestyle=:dash)
end
vlines!(ax3, [1.0, 4.0], color=:gray, linestyle=:dot)
text!(ax3, 1.05, 0.5, text="refine", fontsize=10, color=:gray, space=:data)
text!(ax3, 4.05, 0.5, text="derefine", fontsize=10, color=:gray, space=:data)
axislegend(ax3, "dashed: zero rate", position=:rc, framevisible=false, titlesize=10)

mkpath("validation/figures")
save("validation/figures/dust_growth_tests.png", fig, px_per_unit=2)
println("wrote validation/figures/dust_growth_tests.png")
