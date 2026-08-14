# Static shwave test on the two-level shear mesh (doc fig, sec. 10.4):
# (1) the sheared-mode density on the SMR mesh, (2) where the potential error lives
# (the refinement interfaces at x = +-1/4 — NOT the shear faces at x = +-1/2),
# (3) the SAME error metric on the uniform Phase-1 run (remap truncation only,
# ~1e-6: three decades below the interface/eigenvalue structure), (4) error vs
# resolution at both shear phases.
# Data: validation/run/{bin,log}_p2a_*, mgsw_64 (notebook secs. 9 and 12 runs).
using CairoMakie, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const HERE = @__DIR__
const VAL = dirname(HERE)
include(joinpath(VAL, "..", "scripts", "athenak_bin.jl"))
const RUN = joinpath(VAL, "run")

lastbin(bn, var) = joinpath(RUN, "bin", sort(filter(f -> startswith(f, bn * "." * var),
                                                   readdir(joinpath(RUN, "bin"))))[end])

# closed-form reference: the root-grid discrete-eigenvalue mode (pgen convention;
# qomt = q*Omega*time0 folds into the x wavenumber via sfac)
const N1 = N2 = N3 = 1
const AMP = 0.1
const QOMT = 1.5 * 0.37          # qshear*omega0*time0, time0 < Tshear = 2/3

modearg(x, y, z, lx, ly, lz) = 2pi * (N1 * x / lx + N2 * (y + QOMT * x) / ly + N3 * z / lz)

# per-block z~0 slices of log10 |phi - ref|/phi_amp (and optionally rho - rho0)
function errslices(bnphi; dens_bn=nothing)
    fp = read_bin(lastbin(bnphi, "grav_phi"))
    fw = dens_bn === nothing ? nothing : read_bin(lastbin(dens_bn, "hydro_w"))
    lx = fp.x1max - fp.x1min; ly = fp.x2max - fp.x2min; lz = fp.x3max - fp.x3min
    sfac = QOMT * lx / ly
    kxdx = (N1 + sfac * N2) * 2pi / fp.Nx1
    dd = (2cos(kxdx) - 2) / (lx / fp.Nx1)^2 +
         (2cos(N2 * 2pi / fp.Nx2) - 2) / (ly / fp.Nx2)^2 +
         (2cos(N3 * 2pi / fp.Nx3) - 2) / (lz / fp.Nx3)^2
    phi_amp = AMP / dd                                    # rho0=1, 4piG=1
    nx = fp.nx_mb
    sn = 0.0; sa = 0.0; nc = 0                            # exact means of both fields
    for m in 1:fp.n_mbs
        g = fp.mb_geometry[m, :]; h = (g[2] - g[1]) / nx[1]
        blk = fp.mb_data["grav_phi"][m]
        for k in 1:nx[3], j in 1:nx[2], i in 1:nx[1]
            x = g[1] + (i-0.5)h; y = g[3] + (j-0.5)h; z = g[5] + (k-0.5)h
            sn += blk[i, j, k]; sa += phi_amp * cos(modearg(x, y, z, lx, ly, lz)); nc += 1
        end
    end
    mn = sn / nc; ma = sa / nc
    dens = Tuple{NTuple{4,Float64},Int,Matrix{Float64}}[]
    errs = Tuple{NTuple{4,Float64},Int,Matrix{Float64}}[]
    for m in 1:fp.n_mbs
        g = fp.mb_geometry[m, :]; h = (g[2] - g[1]) / nx[1]
        (g[5] <= 0 < g[6]) || continue
        k = clamp(floor(Int, -g[5] / h) + 1, 1, nx[3])
        lev = fp.mb_logical[m, 4]
        if fw !== nothing
            push!(dens, ((g[1], g[2], g[3], g[4]), lev,
                         fw.mb_data["dens"][m][:, :, k] .- 1.0))
        end
        pblk = fp.mb_data["grav_phi"][m]
        e = zeros(nx[1], nx[2])
        for j in 1:nx[2], i in 1:nx[1]
            x = g[1] + (i-0.5)h; y = g[3] + (j-0.5)h; z = g[5] + (k-0.5)h
            e[i, j] = abs((pblk[i, j, k] - mn) -
                          (phi_amp * cos(modearg(x, y, z, lx, ly, lz)) - ma)) / abs(phi_amp)
        end
        push!(errs, ((g[1], g[2], g[3], g[4]), lev, log10.(max.(e, 1e-16))))
    end
    dens, errs
end
dens, errs_smr = errslices("p2a_64"; dens_bn="p2a_64")
_, errs_uni = errslices("mgsw_64")                       # Phase-1 uniform cache

# error vs resolution from the cached logs
shw(bn) = begin
    t = read(joinpath(RUN, "log_" * bn * ".txt"), String)
    m = match(r"SHWAVE ERROR: max_rel=\s*(\S+)\s*l2_rel=\s*(\S+)", t)
    (parse(Float64, m.captures[1]), parse(Float64, m.captures[2]))
end
Ns = [32, 64, 128]
e_t0 = [shw(bn) for bn in ("p2a_t0_32", "p2a_t0", "p2a_t0_128")]
e_gp = [shw(bn) for bn in ("p2a_32", "p2a_64", "p2a_128")]

# ---- assemble ------------------------------------------------------------------------
fig = Figure(size=(1850, 480))
hms = Any[]
panels = [("gas density  ρ − ρ₀", dens, :RdBu, (-AMP, AMP)),
          ("log₁₀ |Φ − Φ_ref| / Φ_amp  (two-level)", errs_smr, :viridis, (-6.0, -2.0)),
          ("log₁₀ |Φ − Φ_ref| / Φ_amp  (uniform)", errs_uni, :viridis, (-6.0, -2.0))]
for (p, (ttl, ss, cm, cr)) in enumerate(panels)
    ax = Axis(fig[1, p]; xlabel="x", ylabel=p == 1 ? "y" : "", title=ttl,
              aspect=DataAspect())
    for ((x0, x1, y0, y1), lev, m) in ss
        n1, n2 = size(m)
        xs = range(x0 + (x1-x0)/2n1, x1 - (x1-x0)/2n1, length=n1)
        ys = range(y0 + (y1-y0)/2n2, y1 - (y1-y0)/2n2, length=n2)
        push!(hms, heatmap!(ax, xs, ys, m; colorrange=cr, colormap=cm))
        p < 3 && lines!(ax, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0];
                        color=(:white, lev > 0 ? 0.5 : 0.25), linewidth=0.4)
    end
    p < 3 && vlines!(ax, [-0.25, 0.25]; color=:white, linewidth=1.2, linestyle=:dash)
end
Colorbar(fig[2, 1], hms[1]; vertical=false, flipaxis=false)
Colorbar(fig[2, 2:3], hms[end]; vertical=false, flipaxis=false)
ax4 = Axis(fig[1, 4]; xlabel="N (root cells per side)", ylabel="relative Φ error",
           xscale=log2, yscale=log10, title="convergence (two-level mesh)",
           xticks=Ns)
scatterlines!(ax4, Ns, [e[1] for e in e_gp]; color=:crimson, marker=:circle,
              label="max, generic phase")
scatterlines!(ax4, Ns, [e[1] for e in e_t0]; color=:crimson, marker=:utriangle,
              linestyle=:dash, label="max, qomt = 0")
scatterlines!(ax4, Ns, [e[2] for e in e_gp]; color=:steelblue, marker=:circle,
              label="L2, generic phase")
scatterlines!(ax4, Ns, [e[2] for e in e_t0]; color=:steelblue, marker=:utriangle,
              linestyle=:dash, label="L2, qomt = 0")
lines!(ax4, Ns, e_gp[1][1] .* (Ns ./ 32) .^ -1; color=(:gray, 0.7), linestyle=:dot)
lines!(ax4, Ns, e_gp[1][2] .* (Ns ./ 32) .^ -2; color=(:gray, 0.7), linestyle=:dot)
text!(ax4, 90, e_gp[1][1] * (90/32)^-1 * 1.25; text="∝ N⁻¹", color=:gray, fontsize=12)
text!(ax4, 90, e_gp[1][2] * (90/32)^-2 * 0.55; text="∝ N⁻²", color=:gray, fontsize=12)
axislegend(ax4; position=:lb, framevisible=false, labelsize=11)
save(joinpath(HERE, "mg2a_shwave.png"), fig)
println("wrote figures/mg2a_shwave.png")
umax = maximum(maximum(m) for (_, _, m) in errs_uni)
@printf("SMR interface peak %.3e vs uniform max 10^%.2f = %.1e (remap truncation only)\n",
        e_gp[2][1], umax, 10.0^umax)
fig
