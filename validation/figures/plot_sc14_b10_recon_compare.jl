# SC14 beta = 10, standard resolution, hse_outflow z boundary: the same run with different
# hydro reconstructions (plm = the archived gt_sc14_stndrd_b10_hse; ppm4; wenoz+fofc when
# present).  Runs that are not present locally are skipped.
#   sc14_b10_recon_history.png    history (0.25/Omega): M, Q, alpha'/eq.21, alpha_grav and
#                                 alpha_Rey (eq. 19), rms dv and <cs>_rho, dt, energy budget,
#                                 rho_max from the dumps; 10/Omega running means
#   sc14_b10_recon_collapse.png   the cold-start collapse, t = 0-60, unsmoothed
#   sc14_b10_recon_structure.png  t = 200-300 averages of the dump diagnostics: vertical
#                                 profiles, Sigma and kinetic-energy ky spectra and their
#                                 ratios to plm, the stress profiles
#   sc14_b10_recon_maps.png       Sigma(x, y) at matched times
# Inputs per run: GravitoTurb.user.hst and recon_compare.npz from
#   python3 scripts/analysis/gt_recon_compare_extract.py <run> -o <run>/recon_compare.npz
# Usage: julia validation/figures/plot_sc14_b10_recon_compare.jl
using CairoMakie, Printf, DelimitedFiles, Statistics, NPZ
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))

const RUN = joinpath(dirname(@__DIR__), "run")
const GAMMA = 5/3
const BETA = 10.0
const G = 4.25231/(4π)
const AREA = 64.0*64.0
const TCORR = 10.0
eq21(b) = 4/(9*GAMMA*(GAMMA-1))/b
const WIN = [(100.0, 200.0), (200.0, 300.0)]

const RUNS = [("plm",        "gt_sc14_stndrd_b10_hse",        :black),
              ("ppm4",       "gt_sc14_stndrd_b10_hse_ppm4",   RGBf(0.80, 0.14, 0.22)),
              ("wenoz+fofc", "gt_sc14_stndrd_b10_hse_wzfofc", RGBf(0.17, 0.42, 0.72))]

function read_hist(dir)
    d = readdlm(joinpath(RUN, dir, "GravitoTurb.user.hst"), comments=true, comment_char='#')
    keep = falses(size(d, 1)); tmin = Inf                 # drop restart overlaps
    for i in size(d, 1):-1:1
        d[i, 1] < tmin && (keep[i] = true; tmin = d[i, 1])
    end
    d = d[keep, :]
    M = d[:, 3]
    (t=d[:, 1], dt=d[:, 2], M=M, cs=d[:, 4] ./ M,
     Q=(d[:, 4] ./ M) ./ (π*G .* M ./ AREA),
     ag=d[:, 5] ./ d[:, 7], ar=d[:, 6] ./ d[:, 7],
     ab=(2/3) .* (d[:, 8] .+ d[:, 9]) ./ d[:, 10] ./ eq21(BETA),
     dv=sqrt.(d[:, 11] ./ M),
     hc=1.5 .* (d[:, 8] .+ d[:, 9]) ./ (d[:, 12] ./ BETA))   # stress heating / cooling
end

function runmean(t, y, w)
    out = similar(y)
    for i in eachindex(t)
        m = abs.(t .- t[i]) .<= w/2
        out[i] = mean(y[m])
    end
    out
end

sem(t, y, w) = (m = (t .>= w[1]) .& (t .<= w[2]);
                (mean(y[m]), std(y[m])/sqrt(max(1.0, (w[2]-w[1])/TCORR))))

R = []
for (lab, dir, col) in RUNS
    isfile(joinpath(RUN, dir, "GravitoTurb.user.hst")) || (println("  (skipping $lab: $dir not present)"); continue)
    h = read_hist(dir)
    f = joinpath(RUN, dir, "recon_compare.npz")
    z = isfile(f) ? npzread(f) : nothing
    push!(R, (lab=lab, dir=dir, col=col, h=h, z=z))
end

# ---------------------------------------------------------------- table
println("\nSaturated state (mean +- s.e.m. with a $(TCORR)/Omega correlation time):")
for w in WIN
    @printf("\n  t = %.0f-%.0f\n  %-11s %-15s %-13s %-13s %-13s %-8s %-7s %-7s %-9s %-9s\n", w..., "run",
            "alpha'/eq21", "a19 x1e-2", "grav x1e-2", "Rey x1e-2", "Q", "dv", "<cs>", "heat/cool", "dlnM/dt")
    for r in R
        h = r.h
        h.t[end] < w[2] - 1 && @printf("  (%s ends at t = %.1f)\n", r.lab, h.t[end])
        (ab, abe) = sem(h.t, h.ab, w); (g, ge) = sem(h.t, h.ag, w); (re, ree) = sem(h.t, h.ar, w)
        (a19, a19e) = sem(h.t, h.ag .+ h.ar, w)
        m = (h.t .>= w[1]) .& (h.t .<= w[2])
        dlnM = (h.M[m][end] - h.M[m][1])/(h.t[m][end] - h.t[m][1])/mean(h.M[m])
        @printf("  %-11s %.3f+-%.3f   %.2f+-%.2f     %.2f+-%.2f     %+.2f+-%.2f    %.3f    %.2f    %.2f    %.3f     %+.2e\n",
                r.lab, ab, abe, 100a19, 100a19e, 100g, 100ge, 100re, 100ree,
                mean(h.Q[m]), mean(h.dv[m]), mean(h.cs[m]), mean(h.hc[m]), dlnM)
    end
end
println("\nCollapse:")
for r in R
    h = r.h; m = h.t .<= 60
    i = argmin(h.dt[m]); j = argmin(h.Q[m])
    k30 = argmin(abs.(h.t .- 30))
    zs = r.z === nothing ? "" : @sprintf("  rho_max(t=20) = %.1f, (t=30) = %.1f", r.z["rho_max"][3], r.z["rho_max"][4])
    @printf("  %-11s min dt %.2e at t = %.1f;  min Q %.3f at t = %.1f;  M(30)/M(0) = %.4f%s\n",
            r.lab, h.dt[m][i], h.t[m][i], h.Q[m][j], h.t[m][j], h.M[k30]/h.M[1], zs)
end

# ---------------------------------------------------------------- history figure
function hax(gp, ylab; yscale=identity, xl=(0, 300), title="")
    Axis(gp; xlabel="Ω t", ylabel=ylab, yscale=yscale, title=title, titlealign=:left,
         titlefont=:bold, xgridcolor=(:gray, 0.15), ygridcolor=(:gray, 0.15),
         limits=(xl, nothing))
end
fig = Figure(size=(1250, 1150))
Label(fig[0, 1:2], "SC14 β = 10, 4 cells/H, hse_outflow: reconstruction comparison (history at 0.25/Ω, 10/Ω running means)",
      fontsize=16, font=:bold)
a = [hax(fig[1, 1], "M / M(0)", title="(a) mass"),
     hax(fig[1, 2], "Q", title="(b) Toomre Q"),
     hax(fig[2, 1], "α′ / eq. 21", title="(c) volume-averaged stress vs Gammie"),
     hax(fig[2, 2], "α (eq. 19)", title="(d) gravitational (solid), Reynolds (dashed)"),
     hax(fig[3, 1], "rms δv,  ⟨cs⟩_ρ", title="(e) rms δv (solid), ⟨cs⟩_ρ (dashed)"),
     hax(fig[3, 2], "dt", yscale=log10, title="(f) time step"),
     hax(fig[4, 1], "stress heating / cooling", title="(g) energy budget"),
     hax(fig[4, 2], "ρ_max", yscale=log10, title="(h) ρ_max (dumps, every 10/Ω)")]
for r in R
    h = r.h; c = r.col
    sm(y) = runmean(h.t, y, TCORR)
    lines!(a[1], h.t, h.M ./ h.M[1], color=c, label=r.lab)
    lines!(a[2], h.t, sm(h.Q), color=c)
    lines!(a[3], h.t, sm(h.ab), color=c)
    lines!(a[4], h.t, sm(h.ag), color=c)
    lines!(a[4], h.t, sm(h.ar), color=c, linestyle=:dash)
    lines!(a[5], h.t, sm(h.dv), color=c)
    lines!(a[5], h.t, sm(h.cs), color=c, linestyle=:dash)
    lines!(a[6], h.t, h.dt, color=(c, 0.8), linewidth=0.8)
    lines!(a[7], h.t, sm(h.hc), color=c)
    r.z === nothing || scatterlines!(a[8], r.z["time"], r.z["rho_max"], color=c, markersize=6)
end
hlines!(a[3], [1.0], color=:gray50, linestyle=:dot); hlines!(a[7], [1.0], color=:gray50, linestyle=:dot)
ylims!(a[3], 0.4, 1.6); ylims!(a[7], 0.4, 1.6)
for ax in a[[3, 4, 7]], w in WIN
    vspan!(ax, w..., color=(:gray, 0.06))
end
axislegend(a[1], position=:rt, framevisible=false)
save(joinpath(@__DIR__, "sc14_b10_recon_history.png"), fig; px_per_unit=2)

# ---------------------------------------------------------------- collapse figure
fig = Figure(size=(1250, 380))
c = [hax(fig[1, 1], "Q", xl=(0, 60), title="(a) Q"),
     hax(fig[1, 2], "dt", yscale=log10, xl=(0, 60), title="(b) time step"),
     hax(fig[1, 3], "M / M(0)", xl=(0, 60), title="(c) mass"),
     hax(fig[1, 4], "α (eq. 19)", xl=(0, 60), title="(d) total α, unsmoothed")]
for r in R
    h = r.h; m = h.t .<= 60
    lines!(c[1], h.t[m], h.Q[m], color=r.col, label=r.lab)
    lines!(c[2], h.t[m], h.dt[m], color=r.col)
    lines!(c[3], h.t[m], h.M[m] ./ h.M[1], color=r.col)
    lines!(c[4], h.t[m], h.ag[m] .+ h.ar[m], color=r.col)
end
axislegend(c[1], position=:rt, framevisible=false)
save(joinpath(@__DIR__, "sc14_b10_recon_collapse.png"), fig; px_per_unit=2)

# ---------------------------------------------------------------- structure figure
RZ = filter(r -> r.z !== nothing, R)
tavg(z, k, w=WIN[2]) = (m = (z["time"] .>= w[1]) .& (z["time"] .<= w[2]); vec(mean(z[k][m, :], dims=1)))
fig = Figure(size=(1300, 900))
Label(fig[0, 1:4], "SC14 β = 10: structure averaged over t = 200–300 (dumps every 10/Ω)", fontsize=16, font=:bold)
pz(gp, xl, tt; xscale=identity) = Axis(gp; xlabel=xl, ylabel="z / H", title=tt, titlealign=:left,
                                       titlefont=:bold, xscale=xscale, xgridcolor=(:gray, 0.15), ygridcolor=(:gray, 0.15))
b = [pz(fig[1, 1], "⟨ρ⟩", "(a) density", xscale=log10),
     pz(fig[1, 2], "⟨cs²⟩_ρ", "(b) sound speed²"),
     pz(fig[1, 3], "⟨ρ vx δvy⟩", "(c) Reynolds stress"),
     pz(fig[1, 4], "⟨ρ vz⟩", "(d) vertical mass flux")]
for r in RZ
    z = r.z; zc = vec(z["zc"])
    lines!(b[1], tavg(z, "rho_z"), zc, color=r.col, label=r.lab)
    lines!(b[2], tavg(z, "cs2_z"), zc, color=r.col)
    lines!(b[3], tavg(z, "wrey_z"), zc, color=r.col)
    lines!(b[4], tavg(z, "mflux_z"), zc, color=r.col)
end
vlines!(b[3], [0.0], color=:gray60, linestyle=:dot); vlines!(b[4], [0.0], color=:gray60, linestyle=:dot)
axislegend(b[1], position=:rt, framevisible=false)

sp(gp, yl, tt; ys=log10) = Axis(gp; xlabel="k_y H", ylabel=yl, xscale=log10, yscale=ys, title=tt,
                                titlealign=:left, titlefont=:bold, xgridcolor=(:gray, 0.15), ygridcolor=(:gray, 0.15))
s = [sp(fig[2, 1], "P_Σ(k_y)", "(e) Σ spectrum along y"),
     sp(fig[2, 2], "ratio to plm", "(f) Σ spectrum / plm", ys=identity),
     sp(fig[2, 3], "E_K(k_y), |z| < H", "(g) kinetic-energy spectrum"),
     sp(fig[2, 4], "ratio to plm", "(h) E_K / plm", ys=identity)]
ky = vec(RZ[1].z["ky"])[2:end]
base = isempty(RZ) ? nothing : RZ[1]
for r in RZ
    py = tavg(r.z, "py")[2:end]; ke = tavg(r.z, "ke_y")[2:end]
    lines!(s[1], ky, py, color=r.col); lines!(s[3], ky, ke, color=r.col)
    lines!(s[2], ky, py ./ tavg(base.z, "py")[2:end], color=r.col)
    lines!(s[4], ky, ke ./ tavg(base.z, "ke_y")[2:end], color=r.col)
end
for ax in s[[2, 4]]
    hlines!(ax, [1.0], color=:gray60, linestyle=:dot)
    vlines!(ax, [2π/(4*64/256)], color=:gray60, linestyle=:dash)     # 4-cell wavelength
end
# stress profiles, t = 200-300 (gravitational part needs grav_phi at every dump there)
w3 = Axis(fig[3, 1:2]; xlabel="stress", ylabel="z / H", title="(i) ⟨g_x g_y⟩/4πG (solid), ⟨ρ vx δvy⟩ (dashed), t = 200–300",
          titlealign=:left, titlefont=:bold)
ts = Axis(fig[3, 3:4]; xlabel="Ω t", ylabel="value", yscale=log10,
          title="(j) δΣ rms/mean (solid), SC14 eq. 17 δρ rms (dashed)", titlealign=:left, titlefont=:bold)
for r in RZ
    z = r.z; zc = vec(z["zc"])
    wg = tavg(z, "wgrv_z")
    any(isnan, wg) ? println("  ($(r.lab): grav_phi missing in t = 200-300, no w_grv profile)") :
                     lines!(w3, wg, zc, color=r.col)
    lines!(w3, tavg(z, "wrey_z"), zc, color=r.col, linestyle=:dash)
    lines!(ts, z["time"][2:end], z["sigma_rms"][2:end], color=r.col)
    lines!(ts, z["time"][2:end], z["drho_rms"][2:end], color=r.col, linestyle=:dash)
end
vlines!(w3, [0.0], color=:gray60, linestyle=:dot)
save(joinpath(@__DIR__, "sc14_b10_recon_structure.png"), fig; px_per_unit=2)

# ---------------------------------------------------------------- Sigma maps
TM = [20.0, 30.0, 100.0, 300.0]
fig = Figure(size=(300*length(TM) + 120, 300*length(RZ) + 60))
Label(fig[0, 1:length(TM)], "Σ_g(x, y), SC14 β = 10 (log₁₀ Σ)", fontsize=16, font=:bold)
hm = nothing
for (row, r) in enumerate(RZ), (col, tm) in enumerate(TM)
    z = r.z; i = argmin(abs.(z["time"] .- tm))
    S = Float64.(z["sigma"][i, :, :])                # (y, x)
    x = range(z["x1min"], z["x1max"], length=size(S, 2)); y = range(z["x2min"], z["x2max"], length=size(S, 1))
    ax = Axis(fig[row, col], aspect=DataAspect(), title=@sprintf("%s, Ω t = %.0f", r.lab, z["time"][i]),
              xticklabelsvisible=row == length(RZ), yticklabelsvisible=col == 1)
    global hm = heatmap!(ax, x, y, log10.(S'), colormap=:cividis, colorrange=(-0.8, 1.0))
end
Colorbar(fig[1:length(RZ), length(TM)+1], hm, label="log₁₀ Σ")
save(joinpath(@__DIR__, "sc14_b10_recon_maps.png"), fig; px_per_unit=2)
println("\nok")
