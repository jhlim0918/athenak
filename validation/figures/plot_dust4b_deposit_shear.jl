# Phase 4b (i): convergence of the shear-periodic additive deposit fold
# (pgen dust_deposit_shear, inputs/dust/dust_deposit_shear.athinput) at a fixed
# half-cell fractional shift (yshear/dy = nx2/2 + 1/2) for the three y-remap orders.
# Left: L1 relative error of the received strips vs. the shifted analytic expectation;
# right: max relative error.  Second-order guide anchored on the donor-cell L1 point.
#
# Data: validation/run/dust4b/deposit_test/deposit_shear_results.txt (README there)
# Usage: julia validation/figures/plot_dust4b_deposit_shear.jl
using CairoMakie, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))

const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run", "dust4b", "deposit_test")

# parse the "fixed fractional shift" section
lines = readlines(joinpath(RUN, "deposit_shear_results.txt"))
i0 = findfirst(l -> occursin("fixed fractional shift", l), lines)
i1 = findfirst(l -> occursin("## MPI", l), lines)
data = Dict{String,Vector{NTuple{3,Float64}}}()
for l in lines[i0+1:i1-1]
    m = match(r"nx2= (\d+) .* remap= (\w+) .* max_rel= ([\d.e+-]+) l1_rel= ([\d.e+-]+)", l)
    m === nothing && continue
    push!(get!(data, m.captures[2], NTuple{3,Float64}[]),
          (parse(Float64, m.captures[1]), parse(Float64, m.captures[3]), parse(Float64, m.captures[4])))
end

fig = Figure(size=(1000, 430))
ax1 = Axis(fig[1, 1], xscale=log2, yscale=log10, xlabel="cells in y (nx2)", ylabel="L1 relative error",
           title="deposit fold: strips vs. shifted expectation (eps = 1/2)")
ax2 = Axis(fig[1, 2], xscale=log2, yscale=log10, xlabel="cells in y (nx2)", ylabel="max relative error",
           title="max norm")
for (name, lab) in (("dc", "donor cell"), ("plm", "PLM (default)"), ("ppmx", "PPMX"))
    haskey(data, name) || continue
    d = sort(data[name])
    n = [x[1] for x in d]; mx = [x[2] for x in d]; l1 = [x[3] for x in d]
    scatterlines!(ax1, n, l1, label=lab, markersize=12)
    scatterlines!(ax2, n, mx, label=lab, markersize=12)
    o = log.(l1[1:end-1] ./ l1[2:end]) ./ log.(n[2:end] ./ n[1:end-1])
    println(lab, "  L1 orders: ", round.(o, digits=2), "  max orders: ",
            round.(log.(mx[1:end-1] ./ mx[2:end]) ./ log.(n[2:end] ./ n[1:end-1]), digits=2))
end
d = sort(data["dc"]); n = [x[1] for x in d]
lines!(ax1, n, d[1][3] .* (n[1] ./ n).^2, color=:black, linestyle=:dash, label="2nd order")
lines!(ax2, n, d[1][2] .* (n[1] ./ n).^2, color=:black, linestyle=:dash, label="2nd order")
axislegend(ax1, position=:lb, framevisible=false)
axislegend(ax2, position=:lb, framevisible=false)
save(joinpath(HERE, "dust4b_deposit_shear.png"), fig, px_per_unit=2)
println("wrote dust4b_deposit_shear.png")
fig
