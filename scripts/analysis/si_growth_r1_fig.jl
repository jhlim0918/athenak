# R1 figure: the streaming instability with live dust growth (thin 3D box, quasi-2D).
#   julia scripts/analysis/si_growth_r1_fig.jl <fid run dir> [<nogrowth run dir>] [out.png]
# Run directories hold r1.phst, vtk/r1.dust_dpm.*.vtk and pvtk/r1.prtcl_all.*.part.vtk.
#   (a) max rho_p vs t (with the Hill density 180 rho_g0 of Lim et al. 2025), fid vs nogrowth
#   (b) Stokes numbers vs t: mass-weighted mean, max, min
#   (c) Stokes-number distributions (particle mass fractions) at several times
#   (d) rho_p(x, z) at the last particle snapshot (y-averaged PM density, cividis)
#   (e) the particles in (x, z) at that time, coloured by their Stokes number
using CairoMakie, Printf

fid = length(ARGS) > 0 ? ARGS[1] : "validation/run/r1/fid"
ctl = length(ARGS) > 1 ? ARGS[2] : ""
out = length(ARGS) > 2 ? ARGS[3] : joinpath(fid, "r1_growth_si.png")
rho_hill = 180.0

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

# read a big-endian Float32/Float64 block of n values starting at byte offset off (0-based)
function be_block(data, off, n, T)
    v = reinterpret(T, data[off+1:off+n*sizeof(T)])
    return Float64.(ntoh.(v))
end

function header_time(data)
    m = match(r"time=\s*([-+0-9.eE]+)", String(data[1:min(400, length(data))]))
    parse(Float64, m.captures[1])
end

# AthenaK particle VTK: positions, SCALARS (gid, ptag, pspecies, ptaus), VECTORS pvel
function read_pvtk(fn)
    data = read(fn)
    s = String(copy(data))
    t = header_time(data)
    m = match(r"POINTS (\d+) (\w+)\n", s)
    n = parse(Int, m.captures[1])
    T = m.captures[2] == "double" ? Float64 : Float32
    off = m.offset + length(m.match) - 1
    xyz = reshape(be_block(data, off, 3n, T), 3, n)
    out = Dict{String,Any}("time" => t, "xyz" => xyz)
    for mm in eachmatch(r"SCALARS (\S+) \S+\nLOOKUP_TABLE default\n", s)
        o = mm.offset + length(mm.match) - 1
        out[mm.captures[1]] = be_block(data, o, n, T)
    end
    return out
end

# AthenaK mesh VTK (STRUCTURED_POINTS, one cell-centred scalar): array (nx, ny, nz)
function read_vtk_scalar(fn)
    data = read(fn)
    s = String(copy(data))
    t = header_time(data)
    d = parse.(Int, split(match(r"DIMENSIONS (\d+) (\d+) (\d+)", s).match)[2:4]) .- 1
    o = parse.(Float64, split(match(r"ORIGIN (\S+) (\S+) (\S+)", s).match)[2:4])
    h = parse.(Float64, split(match(r"SPACING (\S+) (\S+) (\S+)", s).match)[2:4])
    mm = match(r"SCALARS \S+ (\w+)\nLOOKUP_TABLE default\n", s)
    T = mm.captures[1] == "double" ? Float64 : Float32
    off = mm.offset + length(mm.match) - 1
    a = reshape(be_block(data, off, prod(d), T), d[1], d[2], d[3])
    x = o[1] .+ h[1] .* (collect(1:d[1]) .- 0.5)
    z = o[3] .+ h[3] .* (collect(1:d[3]) .- 0.5)
    return t, x, z, a
end

hf = read_phst(joinpath(fid, "r1.phst"))
hc = isempty(ctl) ? nothing : read_phst(joinpath(ctl, "r1.phst"))
pfiles = sort(filter(f -> endswith(f, ".part.vtk"), readdir(joinpath(fid, "pvtk"), join=true)))
dfiles = sort(filter(f -> occursin("dust_dpm", f), readdir(joinpath(fid, "vtk"), join=true)))

fig = Figure(size=(1500, 820), fontsize=13)
cols = Makie.wong_colors()
top = fig[1, 1] = GridLayout()
bot = fig[2, 1] = GridLayout()

ax1 = Axis(top[1, 1], xlabel="t Ω", ylabel="max ρ_p / ρ_g0", yscale=log10,
           title="(a) maximum particle density")
lines!(ax1, hf["time"], hf["dpm_max"], color=cols[1], label="growth (S0 = 10⁻³)")
if hc !== nothing
    lines!(ax1, hc["time"], hc["dpm_max"], color=cols[2], label="no growth (S = 10⁻³)")
end
hlines!(ax1, [rho_hill], color=:gray, linestyle=:dash, label="ρ_H = 180 ρ_g0")
axislegend(ax1, position=:lt, framevisible=false)

ax2 = Axis(top[1, 2], xlabel="t Ω", ylabel="S = Ω t_s", yscale=log10,
           title="(b) Stokes numbers (growth run)")
lines!(ax2, hf["time"], hf["St_max"], color=cols[3], label="max")
lines!(ax2, hf["time"], hf["St_mean"], color=cols[1], linewidth=2, label="mass-weighted mean")
lines!(ax2, hf["time"], hf["St_min"], color=cols[5], label="min")
axislegend(ax2, position=:rb, framevisible=false)

ax3 = Axis(top[1, 3], xlabel="log₁₀ S", ylabel="mass fraction per 0.05 dex",
           title="(c) Stokes-number distribution (S0 = 10⁻³)")
tmax = header_time(read(pfiles[end]))
want = unique(filter(t -> t <= tmax + 1e-6, [10.0, 20.0, 40.0, 60.0, 100.0, 200.0, tmax]))
edges = range(-4.0, 0.0, length=81)
ts = [header_time(read(f)) for f in pfiles]
for (n, tw) in enumerate(want)
    i = argmin(abs.(ts .- tw))
    p = read_pvtk(pfiles[i])
    ls = log10.(p["ptaus"])
    cnt = [count(x -> edges[k] <= x < edges[k+1], ls) for k in 1:length(edges)-1]
    stairs!(ax3, collect(edges[1:end-1]), cnt ./ length(ls), step=:post,
            color=cgrad(:viridis, length(want), categorical=true)[n],
            label=@sprintf("t = %.0f", ts[i]))
end
axislegend(ax3, position=:lt, framevisible=false)
xlims!(ax3, -3.2, 0.0)

# last snapshot: PM density map and the particles coloured by S
p = read_pvtk(pfiles[end])
di = argmin(abs.([header_time(read(f)) for f in dfiles] .- p["time"]))
td, x, z, a = read_vtk_scalar(dfiles[di])
rho = dropdims(sum(a, dims=2), dims=2) ./ size(a, 2)
ax4 = Axis(bot[1, 1], xlabel="x / H", ylabel="z / H", aspect=DataAspect(),
           title=@sprintf("(d) ρ_p / ρ_g0 at t = %.0f Ω⁻¹ (y-averaged)", td))
hm = heatmap!(ax4, x, z, log10.(max.(rho, 1e-3)), colormap=:cividis,
              colorrange=(-2, log10(max(maximum(rho), 10.0))))
Colorbar(bot[1, 2], hm, label="log₁₀ ρ_p", width=12)
ax5 = Axis(bot[1, 3], xlabel="x / H", ylabel="z / H", aspect=DataAspect(),
           title=@sprintf("(e) particles coloured by S at t = %.0f Ω⁻¹", p["time"]))
sc = scatter!(ax5, p["xyz"][1, :], p["xyz"][3, :], color=log10.(p["ptaus"]),
              colormap=:plasma, markersize=1.2, rasterize=2)
Colorbar(bot[1, 4], sc, label="log₁₀ S", width=12)
zl = min(max(sqrt(sum(abs2, p["xyz"][3, :]) / size(p["xyz"], 2)) * 6, 0.01), x[end])
for ax in (ax4, ax5)
    xlims!(ax, x[1] - 0.5 * (x[2] - x[1]), x[end] + 0.5 * (x[2] - x[1]))
    ylims!(ax, -zl, zl)
end

save(out, fig, px_per_unit=2)
println("wrote ", out)
