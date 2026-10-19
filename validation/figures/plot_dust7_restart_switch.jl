# dust7 test 4: the Godines+26 10 au 64^3 state at 20 P continued to 21 P (no gravity) with
# orbital advection, and after rst_orbital_advection.py without it.
#   (a) max rho_p/rho0   (b) gas kinetic energy in x and z (frame-independent components)
#   (c,d) Sigma_p/Sigma_g0 at the last dump of each run, (e) their difference
#   julia plot_dust7_restart_switch.jl <run_fargo> <run_nofargo> out.png
using CairoMakie, Printf
d1, d2, out = ARGS
P = 2pi
rows(f) = (r = [parse.(Float64, split(l)) for l in eachline(f) if !startswith(l, "#") && !isempty(strip(l))]; reduce(hcat, r)')
ph(d) = rows(first(filter(x -> endswith(x, ".phst"), readdir(d, join=true))))
hs(d) = rows(first(filter(x -> endswith(x, ".hydro.hst"), readdir(d, join=true))))
function lastdpm(dir)
    fs = sort(filter(f -> occursin("dust_dpm", f) && endswith(f, ".vtk"), readdir(joinpath(dir, "vtk"), join=true)))
    io = open(fs[end]); meta = Dict{String,Vector{String}}(); tt = 0.0
    while !haskey(meta, "CELL_DATA")
        l = readline(io); m = match(r"time= *([-+0-9.eE]+)", l); m !== nothing && (tt = parse(Float64, m[1]))
        p = split(l); isempty(p) || (meta[p[1]] = p[2:end])
    end
    n = parse.(Int, meta["DIMENSIONS"]) .- 1; org = parse.(Float64, meta["ORIGIN"]); sp = parse.(Float64, meta["SPACING"])
    l = ""; while !startswith(l, "LOOKUP"); l = readline(io); end
    a = Array{Float32}(undef, n...); read!(io, a); close(io)
    tt/P, org[1] .+ sp[1] .* (0.5:1:n[1]), dropdims(sum(Float64.(ntoh.(a)), dims=3), dims=3) .* sp[3] ./ sqrt(2pi)
end
fig = Figure(size=(1400, 900), fontsize=15)
axa = Axis(fig[1, 1:2], yscale=log10, xlabel=L"t/P", ylabel=L"\max\,\rho_p/\rho_0", title="(a) maximum dust density")
axb = Axis(fig[1, 3:4], yscale=log10, xlabel=L"t/P", ylabel="gas kinetic energy", title="(b) gas KE: x (solid), z (dashed)")
for (d, lab, col) in ((d1, "orbital advection", :black), (d2, "no orbital advection (converted at 20 P)", :crimson))
    a = ph(d); h = hs(d)
    lines!(axa, a[:, 1] ./ P, a[:, 12], color=col, label=lab)
    lines!(axb, h[:, 1] ./ P, h[:, 7], color=col, label=lab); lines!(axb, h[:, 1] ./ P, h[:, 9], color=col, linestyle=:dash)
end
axislegend(axa, position=:lb, labelsize=12)
m1 = lastdpm(d1); m2 = lastdpm(d2)
hi = max(maximum(m1[3]), maximum(m2[3]))
local hm
for (n, (lab, m)) in enumerate(zip(("orbital advection", "no orbital advection"), (m1, m2)))
    ax = Axis(fig[2, n], aspect=DataAspect(), xlabel=L"x/H", ylabel=(n == 1 ? L"y/H" : ""),
              title=@sprintf("(%s) %s, t = %.2f P", "cd"[n], lab, m[1]))
    global hm = heatmap!(ax, m[2], m[2], m[3], colormap=:cividis, colorscale=log10, colorrange=(1e-2, hi))
end
Colorbar(fig[2, 3], hm, label=L"\Sigma_p/\Sigma_{g,0}")
dS = m2[3] .- m1[3]; lim = maximum(abs.(dS))
axe = Axis(fig[2, 4], aspect=DataAspect(), xlabel=L"x/H",
           title=@sprintf("(e) difference (d)-(c), max %.2g\n(rms of (c): %.2g)", lim, sqrt(sum(abs2, m1[3])/length(m1[3]))))
hd = heatmap!(axe, m1[2], m1[2], dS, colormap=:balance, colorrange=(-lim, lim))
Colorbar(fig[2, 5], hd)
save(out, fig)
@printf("wrote %s; Sigma_p corr = %.4f, rel L2 diff = %.3e\n", out,
        sum((m1[3] .- sum(m1[3])/length(m1[3])) .* (m2[3] .- sum(m2[3])/length(m2[3]))) /
        sqrt(sum(abs2, m1[3] .- sum(m1[3])/length(m1[3])) * sum(abs2, m2[3] .- sum(m2[3])/length(m2[3]))),
        sqrt(sum(abs2, dS))/sqrt(sum(abs2, m1[3])))
