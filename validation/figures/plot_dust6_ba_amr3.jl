# Phase 6, multi-level AMR: the JY07 run BA (tau_s = 1, eps = 0.2, 40 eta r = 2 H box,
# 2D r-z) in its saturated state on a multi-level adaptive mesh (DUST6_BA=BA_amr3: root
# 64^2 + 2 levels; BA_amr4 (default): root 32^2 + 3 levels; finest 256^2 = JY07's
# resolution; inputs/dust/streaming_BA_amr{3,4}.athinput).  The particle-mesh
# dust density is shown at the finest resolution with every MeshBlock outlined (root:
# white, then blue, cyan, yellow by level) -- the counterpart of Figure 5 of the 2019
# proposal (Athena++ with three AMR levels).  The title reports the cell count against
# the uniform 256^2 mesh.
# Data: validation/run/dust6/BA_amr3/ (README there); DUST6_RUN=<dir> overrides.
# Usage: julia validation/figures/plot_dust6_ba_amr3.jl [tsnap]
using CairoMakie, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
const RUN  = get(ENV, "DUST6_RUN", joinpath(dirname(HERE), "run", "dust6"))
include(joinpath(dirname(HERE), "..", "scripts", "athenak_bin.jl"))
tsnap = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : Inf
const CASE = get(ENV, "DUST6_BA", "BA_amr4")          # BA_amr3 (three levels) | BA_amr4 (four)
bindir = joinpath(RUN, CASE, "bin"); base = (CASE == "BA_amr3") ? "si_BA3" : "si_BA4"
fs = sort(filter(f -> occursin("$base.dust_dpm.", f), readdir(bindir)))
times = [read_bin(joinpath(bindir, f)).time for f in fs]
k = isfinite(tsnap) ? argmin(abs.(times .- tsnap)) : length(fs)
fd = read_bin(joinpath(bindir, fs[k]))
lmax = maximum(fd.mb_logical[:, 4]); s = 1 << lmax
a = fill(NaN, fd.Nx1*s, fd.Nx2*s); m1, m2 = fd.nx_mb[1], fd.nx_mb[2]
for m in 1:fd.n_mbs
    lx1, lx2, _, lev = fd.mb_logical[m, :]; blk = fd.mb_data["dustdpm"][m][:, :, 1]
    e = 1 << (lmax - lev); e > 1 && (blk = repeat(blk, inner=(e, e)))
    i0, j0 = lx1*m1*e, lx2*m2*e; a[i0+1:i0+m1*e, j0+1:j0+m2*e] = blk
end
xe = range(fd.x1min, fd.x1max, length=fd.Nx1*s+1); ze = range(fd.x2min, fd.x2max, length=fd.Nx2*s+1)
nlev = [sum(fd.mb_logical[:, 4] .== l) for l in 0:lmax]
ncells = sum(nlev) * m1 * m2
nuni = (fd.Nx1*s) * (fd.Nx2*s)
levstr = join(string.(nlev), " + ")
levnames = join(["root"; ["level $l" for l in 1:lmax]], ", ")
eps0 = 0.2
fig = Figure(size=(1000, 900))
ax = Axis(fig[1, 1], xlabel="x  (H)", ylabel="z  (H)", aspect=DataAspect(),
          title=@sprintf("JY07 BA, %d AMR levels, t = %.0f Ω⁻¹:  %s MeshBlocks (%s), %.1f%% of the cells of the uniform 256² mesh",
                         lmax+1, fd.time, levstr, levnames, 100*ncells/nuni))
hm = heatmap!(ax, xe, ze, log10.(max.(a ./ eps0, 1e-2)), colormap=:inferno, colorrange=(-1.7, 0.8))
# outline every MeshBlock, one colour per level (coarse = bold, finest = thin)
lc = (:white, :magenta, :cyan, :springgreen)
lw = (2.2, 1.8, 1.1, 0.7)
for (r, lev) in block_outlines(fd)
    lines!(ax, [r[1], r[2], r[2], r[1], r[1]], [r[3], r[3], r[4], r[4], r[3]],
           color=(lc[lev+1], 0.95), linewidth=lw[lev+1])
end
# legend: only the levels actually present, with their block counts and cell sizes
let hs = [], ls = []
    for l in 0:lmax
        nlev[l+1] > 0 || continue
        push!(hs, LineElement(color=lc[l+1], linewidth=max(lw[l+1], 2.0)))
        cells = fd.Nx1 << l
        push!(ls, @sprintf("level %d: %d blocks (%d² eff.)", l, nlev[l+1], cells))
    end
    axislegend(ax, hs, ls, position=:lb, framevisible=true, framecolor=(:white, 0.4),
               backgroundcolor=(:black, 0.55), labelcolor=:white, labelsize=12,
               patchsize=(26, 12))
end
Colorbar(fig[1, 2], hm, label="log₁₀ ρ_p / ⟨ρ_p⟩", width=14)
out = joinpath(HERE, (CASE == "BA_amr3") ? "dust6_ba_amr3.png" : "dust6_ba_amr4.png")
save(out, fig, px_per_unit=1.6)
println("wrote ", out, "  t=", fd.time, "  blocks per level=", nlev)
