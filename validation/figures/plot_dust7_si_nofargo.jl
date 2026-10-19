# dust7 test 3: the stratified streaming instability (YJ14 box, 80/H, St = 0.3, Z = 0.02,
# Pi = 0.05, Roe) with and without orbital advection.
#   (a) max rho_p/rho0 vs t/P   (b) particle-layer rms thickness sigma_z/H vs t/P
#   (c,d) Sigma_p/Sigma_g0 face-on at the last dust_dpm dump of each run (one colour scale)
#   julia plot_dust7_si_nofargo.jl <run_dir_fargo> <run_dir_nofargo> out.png
using CairoMakie, Printf

d1, d2, out = ARGS
P = 2pi
function phst(dir)
    f = first(filter(x -> endswith(x, ".phst"), readdir(dir, join=true)))
    rows = [parse.(Float64, split(l)) for l in eachline(f) if !startswith(l, "#") && !isempty(strip(l))]
    a = reduce(hcat, rows)'; a[:, 1] ./ P, a[:, 12], a[:, 22]
end
function lastdpm(dir)
    fs = sort(filter(f -> occursin("dust_dpm", f) && endswith(f, ".vtk"), readdir(joinpath(dir, "vtk"), join=true)))
    io = open(fs[end]); meta = Dict{String,Vector{String}}(); tt = 0.0
    while !haskey(meta, "CELL_DATA")
        l = readline(io); m = match(r"time= *([-+0-9.eE]+)", l); m !== nothing && (tt = parse(Float64, m[1]))
        p = split(l); isempty(p) || (meta[p[1]] = p[2:end])
    end
    n = parse.(Int, meta["DIMENSIONS"]) .- 1; org = parse.(Float64, meta["ORIGIN"]); sp = parse.(Float64, meta["SPACING"])
    l = ""; while !startswith(l, "LOOKUP"); l = readline(io); end
    a = Array{Float32}(undef, n...); read!(io, a); close(io); a = Float64.(ntoh.(a))
    S = dropdims(sum(a, dims=3), dims=3) .* sp[3] ./ sqrt(2pi)
    x = org[1] .+ sp[1] .* (0.5:1:n[1]); y = org[2] .+ sp[2] .* (0.5:1:n[2])
    tt/P, x, y, S
end
fig = Figure(size=(1300, 950), fontsize=15)
axa = Axis(fig[1, 1:2], yscale=log10, xlabel=L"t/P", ylabel=L"\max\,\rho_p/\rho_0", title="(a) maximum dust density")
axb = Axis(fig[1, 3:4], yscale=log10, xlabel=L"t/P", ylabel=L"\sigma_z/H", title="(b) particle-layer thickness")
for (d, lab, col) in ((d1, "orbital advection", :black), (d2, "no orbital advection", :crimson))
    t, r, sz = phst(d)
    lines!(axa, t, r, color=col, label=lab); lines!(axb, t, sz, color=col, label=lab)
end
axislegend(axa, position=:lt); axislegend(axb, position=:rt)
maps = [lastdpm(d1), lastdpm(d2)]
lo = 1e-2; hi = maximum(maximum(m[4]) for m in maps)
local hm
for (n, (lab, m)) in enumerate(zip(("orbital advection", "no orbital advection"), maps))
    t, x, y, S = m
    ax = Axis(fig[2, 2n-1:2n], aspect=DataAspect(), xlabel=L"x/H", ylabel=(n == 1 ? L"y/H" : ""),
              title=@sprintf("(%s) %s, t = %.1f P, max %.2f", "cd"[n], lab, t, maximum(S)))
    global hm = heatmap!(ax, x, y, S, colormap=:cividis, colorscale=log10, colorrange=(lo, hi))
end
Colorbar(fig[2, 5], hm, label=L"\Sigma_p/\Sigma_{g,0}")
save(out, fig); println("wrote ", out)
