# Phase 6 (particles + adaptive mesh refinement): the JY07 nonlinear streaming
# instability (run AB: tau_s = 0.1, eps = 1, 2 eta r box, 2D r-z) with AMR on the
# particle-mesh dust density
#   (root 256^2 + one level where rho_p > 8, particles split on refinement) against the
#   uniform 256^2 twin at JY07's resolution.
#   (a) maximum PM dust density vs time (.phst dpm_max), (b) the refined fraction of
#   the AMR mesh vs time (from the dumps), (c) the AMR dust density at the finest
#   resolution with the level-1 MeshBlocks outlined, (d) the uniform 256^2 twin at the
#   same time.
# Data: validation/run/dust6/AB_{u256,amr256}/ (README there); override the data
# home with DUST6_RUN=<dir>.  Usage: julia validation/figures/plot_dust6_si_amr.jl [tsnap]
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))
const HERE = @__DIR__
const RUN  = get(ENV, "DUST6_RUN", joinpath(dirname(HERE), "run", "dust6"))
include(joinpath(dirname(HERE), "..", "scripts", "athenak_bin.jl"))

hst(path) = readdlm(path, comments=true, comment_char='#')
function labels(path)
    for l in eachline(path)
        occursin("[1]=", l) || continue
        return [split(t, "=")[2] for t in split(replace(l, "#" => "")) if occursin("=", t)]
    end
    String[]
end
# the dust density on the finest-level grid: level-1 blocks placed directly, root blocks
# expanded piecewise-constantly by 2 (2D: one z index)
function assemble_finest(fd, var)
    lmax = maximum(fd.mb_logical[:, 4])
    s = 1 << lmax
    out = fill(NaN, fd.Nx1*s, fd.Nx2*s)
    m1, m2 = fd.nx_mb[1], fd.nx_mb[2]
    for m in 1:fd.n_mbs
        lx1, lx2, _, lev = fd.mb_logical[m, :]
        a = fd.mb_data[var][m][:, :, 1]
        e = 1 << (lmax - lev)
        if e > 1
            a = repeat(a, inner=(e, e))
        end
        i0, j0 = lx1*m1*e, lx2*m2*e
        out[i0+1:i0+m1*e, j0+1:j0+m2*e] = a
    end
    any(isnan, out) && error("gaps in the assembled finest grid")
    return out, range(fd.x1min, fd.x1max, length=fd.Nx1*s+1),
           range(fd.x2min, fd.x2max, length=fd.Nx2*s+1)
end
dumps(dir, base) = isdir(joinpath(dir, "bin")) ?
    sort(filter(f -> occursin("$base.dust_dpm.", f) && endswith(f, ".bin"), readdir(joinpath(dir, "bin")))) : String[]

tsnap = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : Inf
cases = (("AB_u256", "si_AB_u256", "uniform 256² (JY07 resolution)"),
         ("AB_amr256", "si_AB_amr256", "AMR: root 256² + level 1 where ρ_p > 8"))
cols = (Makie.wong_colors()[1], Makie.wong_colors()[2])

fig = Figure(size=(1500, 950))
Label(fig[0, 1:2], "JY07 run AB (τₛ = 0.1, ε = 1, 2ηr box, 2D r–z): streaming instability with AMR on the dust density",
      fontsize=16, font=:bold, tellwidth=false)
ax1 = Axis(fig[1, 1], xlabel="t  (Ω⁻¹)", ylabel="max ρ_p / (ε ρ_g)", yscale=log10,
           title="(a) maximum particle-mesh dust density")
for ((d, base, lab), c) in zip(cases, cols)
    f = joinpath(RUN, d, "$base.phst")
    isfile(f) || continue
    h = hst(f); lb = labels(f); col = Dict(l => i for (i, l) in enumerate(lb))
    lines!(ax1, h[:, col["time"]], h[:, col["dpm_max"]], color=c, linewidth=2, label=lab)
end
axislegend(ax1, position=:rb, framevisible=false)

ax2 = Axis(fig[1, 2], xlabel="t  (Ω⁻¹)", ylabel="fraction of the box at level 1",
           title="(b) refined fraction of the AMR mesh")
let d = joinpath(RUN, "AB_amr256"), base = "si_AB_amr256"
    if isdir(joinpath(d, "bin"))
        tt = Float64[]; ff = Float64[]
        for f in dumps(d, base)
            fd = read_bin(joinpath(d, "bin", f))
            nfine = sum(fd.mb_logical[:, 4] .> 0); nroot = sum(fd.mb_logical[:, 4] .== 0)
            push!(tt, fd.time); push!(ff, 0.25*nfine/(nroot + 0.25*nfine))
        end
        lines!(ax2, tt, ff, color=cols[2], linewidth=2)
        scatter!(ax2, tt, ff, color=cols[2], markersize=6)
        ylims!(ax2, 0, 1)
    end
end

# snapshots: the AMR run at the finest resolution and the uniform 256^2 twin
function snapshot!(pos, d, base, title; outline=false)
    fs = dumps(d, base)
    isempty(fs) && return nothing
    times = [read_bin(joinpath(d, "bin", f)).time for f in fs]
    k = isfinite(tsnap) ? argmin(abs.(times .- tsnap)) : length(fs)
    fd = read_bin(joinpath(d, "bin", fs[k]))
    a, xe, ye = assemble_finest(fd, "dustdpm")
    ax = Axis(fig[pos...], xlabel="x  (H)", ylabel="z  (H)", aspect=DataAspect(),
              title=@sprintf("%s, t = %.1f Ω⁻¹", title, fd.time))
    hm = heatmap!(ax, xe, ye, log10.(max.(a, 1e-3)), colormap=:cividis, colorrange=(-1.5, 2))
    if outline
        for (r, lev) in block_outlines(fd)
            lev > 0 || continue
            lines!(ax, [r[1], r[2], r[2], r[1], r[1]], [r[3], r[3], r[4], r[4], r[3]],
                   color=(:magenta, 0.9), linewidth=0.8)
        end
    end
    ax, hm
end
res = snapshot!((2, 1), joinpath(RUN, "AB_amr256"), "si_AB_amr256", "(c) AMR, level-1 blocks outlined"; outline=true)
res2 = snapshot!((2, 2), joinpath(RUN, "AB_u256"), "si_AB_u256", "(d) uniform 256²")
hm = res !== nothing ? res[2] : (res2 !== nothing ? res2[2] : nothing)
hm !== nothing && Colorbar(fig[2, 3], hm, label="log₁₀ ρ_p / (ε ρ_g)", width=14)

save(joinpath(HERE, "dust6_si_amr.png"), fig, px_per_unit=1.5)
println("wrote ", joinpath(HERE, "dust6_si_amr.png"))
