#!/usr/bin/env julia
# Vertical profiles rho(z) and cs(z) of the gravito_turb run, SC14 Figure-1 style:
# thin solid = the static initial condition (the same SC14 eq.-9 hydrostatic
# integration the pgen performs, evaluated analytically here); thick = the
# horizontally averaged profile of the supplied snapshot(s), averaged over all of
# them (faint: each snapshot individually).  Top: rho/rho0 (log); bottom: cs (log).
#
# Usage: julia scripts/plot_gt_zprofiles.jl <out.png> <file1.bin> [file2.bin ...]
# Optional env: CS0 (default 2.12625), FOURPIG (4.25231), GAMMA (5/3),
#   DFLOOR (1e-4) -- the IC constants; PX supersampling (default 2).

using CairoMakie
include(joinpath(@__DIR__, "athenak_bin.jl"))

function ic_profiles(zs; cs0, fourpig, gamma, dfloor, omega0=1.0, rho0=1.0)
    kpoly = cs0^2/(gamma*rho0^(gamma-1))
    zmax = maximum(abs.(zs))
    nfine = 1 << 15
    h = zmax/nfine
    rho_tab = fill(dfloor*rho0, nfine+1)
    rho, mcol = rho0, 0.0
    rho_tab[1] = rho
    drho(z, r, mc) = -r*(omega0^2*z + fourpig*mc)/(gamma*kpoly*r^(gamma-1))
    for l in 1:nfine
        rho <= dfloor*rho0 && break
        z = (l-1)*h
        rmid = max(rho + 0.5h*drho(z, rho, mcol), dfloor*rho0)
        mmid = mcol + 0.5h*rho
        rho = max(rho + h*drho(z + 0.5h, rmid, mmid), dfloor*rho0)
        mcol = mcol + h*rmid
        rho_tab[l+1] = rho
    end
    interp(az) = begin
        r = az/h
        l = clamp(floor(Int, r), 0, nfine-1)
        w = r - l
        (1-w)*rho_tab[l+1] + w*rho_tab[l+2]
    end
    rz = [interp(abs(z)) for z in zs]
    cz = [sqrt(gamma*kpoly*r^(gamma-1)) for r in rz]
    return rz, cz
end

function zprofiles(fname)
    fd = read_bin(fname)
    rho = assemble_root(fd, "dens")
    eint = assemble_root(fd, "eint")
    gamma = parse(Float64, get(ENV, "GAMMA", string(5/3)))
    cs = sqrt.(gamma .* (gamma - 1.0) .* eint ./ rho)
    zs = collect(range(fd.x3min + 0.5*(fd.x3max-fd.x3min)/fd.Nx3,
                       fd.x3max - 0.5*(fd.x3max-fd.x3min)/fd.Nx3, length=fd.Nx3))
    rprof = dropdims(sum(rho, dims=(1,2)), dims=(1,2)) ./ (fd.Nx1*fd.Nx2)
    cprof = dropdims(sum(cs, dims=(1,2)), dims=(1,2)) ./ (fd.Nx1*fd.Nx2)
    om = try parse(Float64, _get_param(fd.header, "<shearing_box>", "omega0"))
    catch; NaN end
    return zs, rprof, cprof, (isnan(om) ? fd.time : om*fd.time)
end

function main()
    out_path = ARGS[1]
    files = ARGS[2:end]
    isempty(files) && error("no bin files given")

    profs = [zprofiles(f) for f in files]
    zs = profs[1][1]
    rmean = reduce(+, (p[2] for p in profs)) ./ length(profs)
    cmean = reduce(+, (p[3] for p in profs)) ./ length(profs)
    tlab = join([string(round(Int, p[4])) for p in profs], ", ")

    cs0 = parse(Float64, get(ENV, "CS0", "2.12625"))
    fourpig = parse(Float64, get(ENV, "FOURPIG", "4.25231"))
    gamma = parse(Float64, get(ENV, "GAMMA", string(5/3)))
    dfloor = parse(Float64, get(ENV, "DFLOOR", "1.0e-4"))
    ric, cic = ic_profiles(zs; cs0, fourpig, gamma, dfloor)

    px = parse(Float64, get(ENV, "PX", "2"))
    fig = Figure(size=(760, 900), backgroundcolor=:white, fontsize=22)
    ax1 = Axis(fig[1, 1]; ylabel="ρ / ρ₀", yscale=log10,
               ylabelsize=26, xticklabelsvisible=false,
               title="vertical profiles: IC vs Ωt = $(tlab)", titlesize=26)
    lines!(ax1, zs, ric, color=:black, linewidth=1.2, label="initial (eq. 9)")
    for p in profs
        lines!(ax1, zs, p[2], color=(:crimson, 0.35), linewidth=1)
    end
    lines!(ax1, zs, rmean, color=:crimson, linewidth=2.5,
           label="gravito-turbulent")
    ylims!(ax1, 5e-5, 2.0)
    axislegend(ax1, position=:ct, framevisible=false, labelsize=18)

    ax2 = Axis(fig[2, 1]; xlabel="z / H", ylabel="cₛ  [H Ω]", yscale=log10,
               xlabelsize=26, ylabelsize=26)
    lines!(ax2, zs, cic, color=:black, linewidth=1.2)
    for p in profs
        lines!(ax2, zs, p[3], color=(:crimson, 0.35), linewidth=1)
    end
    lines!(ax2, zs, cmean, color=:crimson, linewidth=2.5)
    ylims!(ax2, 0.08, 5.0)

    linkxaxes!(ax1, ax2)
    rowgap!(fig.layout, 8)
    save(out_path, fig; px_per_unit=px)
    println("wrote $out_path")
end

main()
