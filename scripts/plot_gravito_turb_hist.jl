#!/usr/bin/env julia
# SC14 diagnostics from the gravito_turb user history file.
#
# Usage: julia scripts/plot_gravito_turb_hist.jl <run_dir> [Lx Ly]
# Reads <run_dir>/GravitoTurb*.user.hst (columns: time dt mass rho_cs rho_wgrv
# rho_wrey rho_prs wgrv wrey rho_cs2 rho_dv2 eint) and writes
# <run_dir>/gravito_turb_hist.png with the Shi & Chiang (2014) Figure-2 panels:
#   top:    stresses alpha (eq. 19, density-weighted; grav + Reynolds components)
#           and alpha' (eq. 20) with the eq.-21 energy-balance line 4/(9g(g-1)beta)
#   middle: rms velocity fluctuation <dv^2>^1/2 / <cs>_rho, and mass/mass0
#   bottom: Toomre Q (eq. 16); SC14's beta=10 plateau Q ~ 1.33 marked.
# Box size defaults to the SC14 half-box (Lx = Ly = 32); pass Lx Ly to override.
# Curves are boxcar-smoothed over 12/Omega as in SC14 Fig. 2 (raw shown faint).
# Constants (G, gamma, beta, Omega) follow the reference input; edit below if the
# run deviates.

using CairoMakie
using DelimitedFiles
using Statistics

const G      = 0.3384
const gamma_ = 5.0/3.0
const beta_  = 10.0
const omega0 = 1.0

function smooth(t, y; width=12.0)
    out = similar(y)
    for i in eachindex(t)
        sel = @. abs(t - t[i]) <= width/2
        out[i] = mean(y[sel])
    end
    return out
end

function main()
    run_dir = rstrip(ARGS[1], '/')
    Lx = length(ARGS) >= 3 ? parse(Float64, ARGS[2]) : 32.0
    Ly = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 32.0

    hst_files = filter(f -> endswith(f, ".user.hst"), readdir(run_dir, join=true))
    isempty(hst_files) && error("no *.user.hst in $run_dir")
    d = readdlm(hst_files[1], comments=true, comment_char='#')

    # restart seams rewind time and append; keep the LAST occurrence of every
    # epoch (reverse scan keeping rows strictly older than everything kept so far)
    keep = falses(size(d, 1))
    tmin = Inf
    for i in size(d, 1):-1:1
        if d[i, 1] < tmin
            keep[i] = true
            tmin = d[i, 1]
        end
    end
    d = d[keep, :]

    t       = d[:, 1]
    mass    = d[:, 3]
    rho_cs  = d[:, 4]
    rho_wg  = d[:, 5]
    rho_wr  = d[:, 6]
    rho_prs = d[:, 7]
    wgrv    = d[:, 8]
    wrey    = d[:, 9]
    rho_cs2 = d[:, 10]
    rho_dv2 = d[:, 11]

    Q      = @. (rho_cs/mass)*omega0/(pi*G*mass/(Lx*Ly))
    a_grav = @. rho_wg/rho_prs
    a_rey  = @. rho_wr/rho_prs
    a_tot  = a_grav .+ a_rey
    a_pr   = @. (2.0/3.0)*(wgrv + wrey)/rho_cs2   # rho_cs2 = gamma*int P
    dv_cs  = @. sqrt(rho_dv2/mass)/(rho_cs/mass)
    a21    = 4.0/(9.0*gamma_*(gamma_ - 1.0))/beta_

    fig = Figure(size=(900, 1000))

    ax1 = Axis(fig[1, 1], ylabel="stress", title="SC14 diagnostics ($(basename(run_dir)))")
    for (y, lbl, col) in ((a_grav, "α gravitational", :crimson),
                          (a_rey,  "α Reynolds",      :steelblue),
                          (a_tot,  "α total (eq. 19)", :black),
                          (a_pr,   "α′ (eq. 20)",     :darkorange))
        lines!(ax1, t, y, color=(col, 0.25), linewidth=1)
        lines!(ax1, t, smooth(t, y), color=col, linewidth=2, label=lbl)
    end
    hlines!(ax1, [a21], color=:darkorange, linestyle=:dash,
            label="eq. 21: 4/(9γ(γ−1))/β")
    hlines!(ax1, [0.0576, 0.0387, 0.0188], color=(:gray, 0.6), linestyle=:dot)
    text!(ax1, t[end]*0.99, 0.0576, text="SC14 tc=10", align=(:right, :bottom),
          color=:gray, fontsize=11)
    ylims!(ax1, -0.02, 0.14)
    axislegend(ax1, position=:rt, framevisible=false, labelsize=11)

    ax2 = Axis(fig[2, 1], ylabel="⟨δv²⟩¹ᐟ² / ⟨cs⟩ρ   and   M/M₀")
    lines!(ax2, t, dv_cs, color=(:seagreen, 0.25), linewidth=1)
    lines!(ax2, t, smooth(t, dv_cs), color=:seagreen, linewidth=2,
           label="δv / cs  (SC14: ~0.82)")
    lines!(ax2, t, mass ./ mass[1], color=:purple, linewidth=2, label="mass / mass₀")
    hlines!(ax2, [1.79/2.18], color=(:gray, 0.6), linestyle=:dot)
    axislegend(ax2, position=:rb, framevisible=false, labelsize=11)

    ax3 = Axis(fig[3, 1], xlabel="Ω t", ylabel="Toomre Q (eq. 16)")
    lines!(ax3, t, Q, color=(:navy, 0.25), linewidth=1)
    lines!(ax3, t, smooth(t, Q), color=:navy, linewidth=2)
    hlines!(ax3, [1.33], color=(:gray, 0.6), linestyle=:dot)
    text!(ax3, t[end]*0.99, 1.33, text="SC14: Q ≈ 1.33", align=(:right, :bottom),
          color=:gray, fontsize=11)
    ylims!(ax3, 0.4, 2.2)

    linkxaxes!(ax1, ax2, ax3)
    out = joinpath(run_dir, "gravito_turb_hist.png")
    save(out, fig)
    println("wrote $out")
end

main()
