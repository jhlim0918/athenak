#!/usr/bin/env julia
# Vertical profile of the gravito-turbulent stress, SC14 Figures 5-6:
#   w_xy(z) = < g_x g_y / (4 pi G)  +  rho v_x dv_y >_xy        (their eq. 19
#             \____gravitational___/   \___Reynolds___/          numerator, NOT
#                                                                normalized by P)
# horizontally averaged at each height and time-averaged over the supplied
# snapshots.  g = -grad(phi) is differenced from the companion grav_phi dump
# (same output index); dv_y is the FARGO-frame azimuthal velocity, already the
# non-Keplerian part.
#
# Their Figure 5 plots <<w_xy(z)>>_t at |z| = 0,1,2,3,4 H against Omega*t_cool
# across the beta scan: this script prints exactly those five numbers per run
# (folded over +/-z), so the figure is assembled by collecting them.
#
# SC14 Fig.-5 reference values at beta = 10 (read off their figure, .hi runs):
#   |z|/H     0      1      2      3      4
#   w_xy   7.5e-2 1.1e-1 3.6e-2 1.65e-2 8e-3
# Note their |z|=1H point sits ABOVE the midplane one: the time-averaged stress
# peaks off the midplane because the midplane Reynolds stress alternates in sign
# and partially cancels in the average -- a distinctive feature to check for.
#
# Usage: julia scripts/plot_gt_stress_z.jl <out.png> <hydro_w1.bin> [more...]
#   Each hydro_w file must have its grav_phi sibling at the same index.
# Optional env: FOURPIG (default 4.25231); PX supersampling (default 2);
#   NOGRAV=1 to plot the Reynolds channel only (no phi dumps needed).

using CairoMakie
using Printf
include(joinpath(@__DIR__, "athenak_bin.jl"))

phi_key(fd) = haskey(fd.mb_data, "grav_phi") ? "grav_phi" : "phi"

# -d(f)/dx on the assembled root grid: centered inside, one-sided at the x edges
# (the shear-periodic wrap is y-offset there; 2 of Nx1 columns, negligible in a
# horizontal average), periodic wrap in y.
function accel(phi, dx, dy)
    nx, ny, nz = size(phi)
    gx = similar(phi); gy = similar(phi)
    @inbounds for k in 1:nz, j in 1:ny
        for i in 2:nx-1
            gx[i,j,k] = -(phi[i+1,j,k] - phi[i-1,j,k])/(2dx)
        end
        gx[1,j,k]  = -(phi[2,j,k]  - phi[1,j,k])/dx
        gx[nx,j,k] = -(phi[nx,j,k] - phi[nx-1,j,k])/dx
    end
    @inbounds for k in 1:nz, i in 1:nx, j in 1:ny
        jm = j == 1  ? ny : j-1
        jp = j == ny ? 1  : j+1
        gy[i,j,k] = -(phi[i,jp,k] - phi[i,jm,k])/(2dy)
    end
    return gx, gy
end

function stress_profile(fname; fourpig, nograv)
    fd = read_bin(fname)
    rho = assemble_root(fd, "dens")
    vx  = assemble_root(fd, "velx")
    vy  = assemble_root(fd, "vely")     # FARGO frame: already the non-Keplerian part
    dx = (fd.x1max-fd.x1min)/fd.Nx1
    dy = (fd.x2max-fd.x2min)/fd.Nx2
    wrey = dropdims(sum(rho.*vx.*vy, dims=(1,2)), dims=(1,2)) ./ (fd.Nx1*fd.Nx2)
    wgrv = zero(wrey)
    if !nograv
        pf = replace(fname, "hydro_w" => "grav_phi")
        isfile(pf) || error("missing companion dump $pf (use NOGRAV=1 to skip)")
        fp = read_bin(pf)
        phi = assemble_root(fp, phi_key(fp))
        gx, gy = accel(phi, dx, dy)
        wgrv = dropdims(sum(gx.*gy, dims=(1,2)), dims=(1,2)) ./ (fd.Nx1*fd.Nx2*fourpig)
    end
    zs = collect(range(fd.x3min + 0.5*(fd.x3max-fd.x3min)/fd.Nx3,
                       fd.x3max - 0.5*(fd.x3max-fd.x3min)/fd.Nx3, length=fd.Nx3))
    om = try parse(Float64, _get_param(fd.header, "<shearing_box>", "omega0"))
         catch; NaN end
    return zs, wgrv, wrey, (isnan(om) ? fd.time : om*fd.time)
end

function main()
    out_path = ARGS[1]
    files = ARGS[2:end]
    isempty(files) && error("no bin files given")
    fourpig = parse(Float64, get(ENV, "FOURPIG", "4.25231"))
    nograv = get(ENV, "NOGRAV", "0") == "1"
    px = parse(Float64, get(ENV, "PX", "2"))

    profs = [stress_profile(f; fourpig, nograv) for f in files]
    zs = profs[1][1]
    wg = reduce(+, (p[2] for p in profs)) ./ length(profs)
    wr = reduce(+, (p[3] for p in profs)) ./ length(profs)
    wt = wg .+ wr
    tlab = join([string(round(Int, p[4])) for p in profs], ", ")

    # SC14 Fig. 5 sampling: fold over +/-z and report at |z| = 0..4 H
    println("# SC14 Fig.-5 values: <<w_xy(|z|)>>_t  [rho0 H^2 Omega^2]")
    println("#  |z|/H      total       gravitational   Reynolds")
    for zt in 0.0:1.0:4.0
        i1 = argmin(abs.(zs .- zt)); i2 = argmin(abs.(zs .+ zt))
        f(v) = 0.5*(v[i1] + v[i2])
        @printf("   %4.1f   %11.4e   %11.4e   %11.4e\n", zt, f(wt), f(wg), f(wr))
    end

    fig = Figure(size=(820, 620), backgroundcolor=:white, fontsize=22)
    ax = Axis(fig[1, 1]; xlabel="z / H", ylabel="⟨w_xy(z)⟩  [ρ₀H²Ω²]",
              xlabelsize=26, ylabelsize=26, titlesize=26,
              title="stress vs height,  Ωt = $(tlab)")
    hlines!(ax, [0.0], color=(:gray, 0.5), linewidth=1)
    lines!(ax, zs, wg, color=:crimson, linewidth=2.5, label="gravitational")
    lines!(ax, zs, wr, color=:steelblue, linewidth=2.5, label="Reynolds")
    lines!(ax, zs, wt, color=:black, linewidth=3, label="total")
    axislegend(ax, position=:rt, framevisible=false, labelsize=19)
    save(out_path, fig; px_per_unit=px)
    println("wrote $out_path")
end

main()
