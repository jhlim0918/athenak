#!/usr/bin/env julia
# Vertical profiles of sound speed and turbulent velocities -- SC14 Figure 7.
# All quantities are horizontally averaged at each height (their eq. 14) and then
# averaged over the supplied snapshots:
#   black dotted : <c_s(z)>
#   red thick    : <dv^2(z)>^1/2  with dv^2 = vx^2 + dvy^2 + vz^2  (their eq. 18)
#   red thin     : the three components separately
#   green        : line-of-sight proxies (their sec. 3.1.3) --
#                  <|v_p(z)|>, the viewing-angle average of the in-plane los
#                  velocity (their eq. 22), which integrates in closed form to
#                      |v_p| = (2/pi) * sqrt(vx^2 + dvy^2),
#                  and <|v_z(z)|>, the face-on proxy.
# The top axis carries the fractional overlying column N(>|z|)/N, so the paper's
# headline claim is readable directly: velocities vary by <~2x from midplane to
# surface while the overlying column drops by orders of magnitude.
#
# dvy is the FARGO-frame azimuthal velocity (already non-Keplerian).
#
# SC14's beta=10 findings this figure should reproduce: dv rises by less than a
# factor of two from midplane to surface; v_z by about five; |v_z|/c_s by ~2.5
# while |v_p|/c_s barely moves; and the roll-over beyond |z| ~ 5H is their
# documented vertical-boundary artifact (their Fig. 8), not physics.
#
# Usage: julia scripts/plot_gt_velocity_z.jl <out.png> <hydro_w1.bin> [more...]
# Optional env: GAMMA (default 5/3); PX supersampling (default 2).

using CairoMakie
include(joinpath(@__DIR__, "athenak_bin.jl"))

function vprofile(fname; gamma)
    fd = read_bin(fname)
    rho  = assemble_root(fd, "dens")
    vx   = assemble_root(fd, "velx")
    vy   = assemble_root(fd, "vely")      # FARGO frame: non-Keplerian already
    vz   = assemble_root(fd, "velz")
    eint = assemble_root(fd, "eint")
    n = fd.Nx1*fd.Nx2
    hav(a) = dropdims(sum(a, dims=(1,2)), dims=(1,2)) ./ n
    cs = sqrt.(gamma .* (gamma - 1.0) .* eint ./ rho)
    zs = collect(range(fd.x3min + 0.5*(fd.x3max-fd.x3min)/fd.Nx3,
                       fd.x3max - 0.5*(fd.x3max-fd.x3min)/fd.Nx3, length=fd.Nx3))
    om = try parse(Float64, _get_param(fd.header, "<shearing_box>", "omega0"))
         catch; NaN end
    return (z = zs,
            cs = hav(cs),
            dv = sqrt.(hav(vx.^2 .+ vy.^2 .+ vz.^2)),
            dvx = sqrt.(hav(vx.^2)), dvy = sqrt.(hav(vy.^2)),
            dvz = sqrt.(hav(vz.^2)),
            vp = hav((2/pi) .* sqrt.(vx.^2 .+ vy.^2)),
            vzabs = hav(abs.(vz)),
            rho = hav(rho),
            t = isnan(om) ? fd.time : om*fd.time)
end

# z positions where the fractional overlying column N(>|z|)/N equals `fracs`
function column_ticks(zs, rhoz, fracs)
    dz = zs[2] - zs[1]
    zpos = Float64[]; labs = String[]
    half = [z for z in zs if z >= 0]
    cum = [sum(rhoz[zs .>= z])*dz for z in half]
    cum ./= cum[1]
    for f in fracs
        i = findfirst(<=(f), cum)
        i === nothing && continue
        for s in (-1, 1)
            push!(zpos, s*half[i])
            push!(labs, f >= 0.1 ? string(f) : (f >= 0.01 ? "0.01" : "0.001"))
        end
    end
    push!(zpos, 0.0); push!(labs, "1")
    p = sortperm(zpos)
    return zpos[p], labs[p]
end

function main()
    out_path = ARGS[1]
    files = ARGS[2:end]
    isempty(files) && error("no bin files given")
    gamma = parse(Float64, get(ENV, "GAMMA", string(5/3)))
    px = parse(Float64, get(ENV, "PX", "2"))

    ps = [vprofile(f; gamma) for f in files]
    zs = ps[1].z
    m(f) = reduce(+, (getproperty(p, f) for p in ps)) ./ length(ps)
    tlab = join([string(round(Int, p.t)) for p in ps], ", ")

    fig = Figure(size=(900, 700), backgroundcolor=:white, fontsize=22)
    ax = Axis(fig[1, 1]; xlabel="z / H", ylabel="velocity  [H Ω]",
              xlabelsize=26, ylabelsize=26)
    lines!(ax, zs, m(:cs), color=:black, linestyle=:dot, linewidth=2.5,
           label="⟨cₛ⟩")
    lines!(ax, zs, m(:dv), color=:crimson, linewidth=3.5, label="⟨δv²⟩¹ᐟ²")
    for (f, ls, lbl) in ((:dvx, :dash, "⟨δvₓ²⟩¹ᐟ²"), (:dvy, :dashdot, "⟨δv_y²⟩¹ᐟ²"),
                         (:dvz, :dot, "⟨v_z²⟩¹ᐟ²"))
        lines!(ax, zs, m(f), color=(:crimson, 0.75), linestyle=ls, linewidth=1.8,
               label=lbl)
    end
    lines!(ax, zs, m(:vp), color=:seagreen, linewidth=2.5, label="⟨|v_p|⟩ (eq. 22)")
    lines!(ax, zs, m(:vzabs), color=:seagreen, linestyle=:dash, linewidth=2.5,
           label="⟨|v_z|⟩")
    Legend(fig[2, 1], ax, orientation=:horizontal, nbanks=2,
           framevisible=false, labelsize=18, colgap=18)

    zpos, labs = column_ticks(zs, m(:rho), [0.5, 0.1, 0.01, 0.001])
    ax2 = Axis(fig[1, 1]; xaxisposition=:top, xlabel="N(>|z|) / N",
               xlabelsize=24, xticks=(zpos, labs), xticklabelsize=17)
    hidespines!(ax2); hideydecorations!(ax2)
    hidexdecorations!(ax2, label=false, ticklabels=false, ticks=false)
    linkxaxes!(ax, ax2)
    Label(fig[0, 1], "vertical velocity profiles,  Ωt = $(tlab)", fontsize=26)
    rowgap!(fig.layout, 4)
    rowsize!(fig.layout, 1, Relative(0.82))
    save(out_path, fig; px_per_unit=px)
    println("wrote $out_path")
end

main()
