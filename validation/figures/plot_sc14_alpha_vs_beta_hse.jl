# Stress vs cooling time for the MORPHED standard-resolution beta ladder on the
# hse_outflow z boundary (gt_sc14_stndrd_b<beta>_hse: beta = 10 from scratch, t = 0-300,
# -> 20 (300-450) -> 40 (450-750) -> 80 (750-1150)), against the same ladder on the
# diode boundary (sc14_alpha_vs_beta_morph.png) and SC14 Table 1.
#   (a) density-weighted alpha = <w_xy>_rho/<P>_rho (SC14 eq. 19): total, gravitational,
#       Reynolds;
#   (b) volume-averaged alpha' = (2/3)<w_xy>/<gamma P> (eq. 20) against eq. 21, with a
#       power-law fit alpha' = A beta^p to the hse points.
# History file only (0.25/Omega); error bars are the s.e.m. with a 10/Omega correlation
# time.  Windows: SC14's trailing 100/Omega (200/Omega at beta >= 40).  The diode
# beta = 80 uses its FIRST 200/Omega (its trailing window is inside the z-boundary mass
# runaway); the hse beta = 80, if present, uses the trailing window like the rest --
# the point of hse_outflow is that it has no runaway, and the printed M change checks it.
# Runs that are not present locally are skipped.
# Usage: julia validation/figures/plot_sc14_alpha_vs_beta_hse.jl
using CairoMakie, Printf, DelimitedFiles, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))

const RUN = joinpath(dirname(@__DIR__), "run")
const GAMMA = 5/3
const TCORR = 10.0
eq21(b) = 4/(9*GAMMA*(GAMMA-1))/b

const HSE = [(10.0, "gt_sc14_stndrd_b10_hse", (200.0, 300.0)),
             (20.0, "gt_sc14_stndrd_b20_hse", (350.0, 450.0)),
             (40.0, "gt_sc14_stndrd_b40_hse", (550.0, 750.0)),
             (80.0, "gt_sc14_stndrd_b80_hse", (950.0, 1150.0))]
const DIODE = [(10.0, "gt_sc14_stndrd_b10_ng4",      (200.0, 300.0)),
               (20.0, "gt_sc14_stndrd_b20_morph",    (350.0, 450.0)),
               (40.0, "gt_sc14_stndrd_b40_morph",    (550.0, 750.0)),
               (80.0, "gt_sc14_stndrd_b80_morphing", (750.0, 950.0))]
# SC14 Table 1: beta => (alpha_Rey, alpha_grav, alpha eq.19, alpha' eq.20) x 1e-2
const SC14_HI = [(4.0,5.69,6.50,12.2,10.0), (5.0,4.19,5.82,10.0,8.19),
                 (8.0,2.06,4.41,6.47,5.10), (10.0,1.70,3.82,5.52,4.06),
                 (20.0,0.46,2.42,2.88,2.02), (40.0,0.27,1.36,1.62,1.03),
                 (80.0,0.35,0.58,0.93,0.52), (120.0,0.14,0.39,0.53,0.31)]
const SC14_STD = [(10.0,1.88,3.87,5.76,4.24), (20.0,0.29,2.55,2.85,2.14),
                  (40.0,-0.06,1.51,1.44,1.06), (80.0,0.13,0.76,0.89,0.55)]

function read_hist(dir)
    files = filter(f -> endswith(f, ".user.hst"), readdir(joinpath(RUN, dir), join=true))
    d = readdlm(files[1], comments=true, comment_char='#')
    keep = falses(size(d, 1)); tmin = Inf
    for i in size(d, 1):-1:1
        d[i, 1] < tmin && (keep[i] = true; tmin = d[i, 1])
    end
    d = d[keep, :]
    (t=d[:, 1], M=d[:, 3], g=d[:, 5] ./ d[:, 7], r=d[:, 6] ./ d[:, 7],
     ab=(2/3) .* (d[:, 8] .+ d[:, 9]) ./ d[:, 10])
end
whist(t, y, w) = (m = (t .>= w[1]) .& (t .<= w[2]);
                  (mean(y[m]), std(y[m])/sqrt(max(1.0, (w[2]-w[1])/TCORR))))

function ladder(runs, tag)
    out = NamedTuple[]
    for (beta, dir, w) in runs
        isdir(joinpath(RUN, dir)) || (println("  (skipping $dir: not present)"); continue)
        h = read_hist(dir)
        w = (w[1], min(w[2], h.t[end]))           # a shorter run averages what it has
        m = (h.t .>= w[1]) .& (h.t .<= w[2])
        (tot, tote) = whist(h.t, h.g .+ h.r, w); (g, ge) = whist(h.t, h.g, w)
        (r, re) = whist(h.t, h.r, w);            (ab, abe) = whist(h.t, h.ab, w)
        dM = h.M[m][end]/h.M[m][1] - 1
        push!(out, (beta=beta, tot=tot, tote=tote, g=g, ge=ge, r=r, re=re, ab=ab, abe=abe))
        @printf("%-6s beta=%3.0f  t=%4.0f-%-4.0f  a19=%6.2f (grav %5.2f, Rey %5.2f)  a'=%5.2f  eq21=%5.2f  a'/eq21=%4.2f  dM/M over window=%+6.3f   [x1e-2]\n",
                tag, beta, w[1], w[2], 100tot, 100g, 100r, 100ab, 100eq21(beta),
                ab/eq21(beta), dM)
    end
    out
end
println("hse_outflow ladder:"); H = ladder(HSE, "hse")
println("diode ladder:");       D = ladder(DIODE, "diode")

# power law alpha' = A beta^p, least squares in log space
function plfit(L)
    x = log.([l.beta for l in L]); y = log.([l.ab for l in L])
    p = cov(x, y)/var(x); A = exp(mean(y) - p*mean(x)); (A, p)
end
(Ah, ph) = plfit(H); (Ad, pd) = plfit(D)
@printf("power-law fit alpha' = A beta^p:  hse A=%.3f p=%.2f   diode A=%.3f p=%.2f   (eq. 21: A=0.40, p=-1)\n",
        Ah, ph, Ad, pd)

const CTOT = :black
const CGRV = RGBf(0.80, 0.14, 0.22)
const CREY = RGBf(0.17, 0.42, 0.72)
bgrid = 10 .^ range(log10(6), log10(140), length=200)
function panel(gp; ylab, title)
    ax = Axis(gp; xscale=log10, yscale=log10, xlabel="Ω t_cool  ( = β )", ylabel=ylab,
              title=title, titlealign=:left, titlefont=:bold,
              xticks=([10, 20, 40, 80, 120], ["10", "20", "40", "80", "120"]),
              yticks=([1e-3, 3e-3, 1e-2, 3e-2, 1e-1], ["10⁻³", "3×10⁻³", "10⁻²", "3×10⁻²", "10⁻¹"]),
              xminorticksvisible=true, yminorticksvisible=true,
              xgridcolor=(:gray, 0.18), ygridcolor=(:gray, 0.18))
    xlims!(ax, 6, 140); ylims!(ax, 1e-3, 1.5e-1)
    ax
end
ok(v) = v > 0
bs(L) = [l.beta for l in L]

fig = Figure(size=(1180, 560))
Label(fig[0, 1:2], "SC14 morphed β-ladder, standard resolution: hse_outflow vs diode z boundary",
      fontsize=18, font=:bold, padding=(0, 0, 4, 0))

ax1 = panel(fig[1, 1]; ylab="α = ⟨w_xy⟩_ρ / ⟨P⟩_ρ   (SC14 eq. 19)",
            title="(a) density-weighted stress")
for (key, ke, col, lab) in ((:tot, :tote, CTOT, "total"), (:g, :ge, CGRV, "gravitational"),
                            (:r, :re, CREY, "Reynolds"))
    for (L, filled, dlab) in ((H, true, "hse"), (D, false, "diode"))
        y = [getfield(l, key) for l in L]; e = [getfield(l, ke) for l in L]; x = bs(L)
        k = ok.(y)
        lines!(ax1, x[k], y[k], color=(col, filled ? 1.0 : 0.45), linewidth=filled ? 2.2 : 1.4,
               linestyle=filled ? :solid : :dash)
        filled && errorbars!(ax1, x[k], y[k], min.(e[k], 0.7 .* y[k]), e[k], color=col,
                             whiskerwidth=8, linewidth=1.3)
        scatter!(ax1, x[k], y[k], color=filled ? col : :white, strokecolor=col,
                 strokewidth=1.6, markersize=filled ? 13 : 11,
                 label=filled ? "hse, $lab" : nothing)
    end
end
scatter!(ax1, [s[1] for s in SC14_STD], [s[4]*1e-2 for s in SC14_STD], color=:white,
         strokecolor=(:black, 0.55), strokewidth=1.4, markersize=11, marker=:rect,
         label="SC14 std-res, total")
axislegend(ax1, position=:lb, framevisible=false, labelsize=11, rowgap=0)

ax2 = panel(fig[1, 2]; ylab="α′ = (2/3)⟨w_xy⟩ / ⟨γP⟩   (SC14 eq. 20)",
            title="(b) volume-averaged stress vs eq. 21")
lines!(ax2, bgrid, eq21.(bgrid), color=:gray40, linestyle=:dash, linewidth=2,
       label="eq. 21: 0.4/β")
lines!(ax2, bgrid, Ah .* bgrid .^ ph, color=CGRV, linewidth=1.6,
       label=@sprintf("fit to hse: %.2f β^%.2f", Ah, ph))
xd = bs(D); yd = [l.ab for l in D]
lines!(ax2, xd, yd, color=(:black, 0.45), linestyle=:dash, linewidth=1.4)
scatter!(ax2, xd, yd, color=:white, strokecolor=:black, strokewidth=1.6, markersize=11,
         label="diode ladder")
xh = bs(H); yh = [l.ab for l in H]; eh = [l.abe for l in H]
errorbars!(ax2, xh, yh, min.(eh, 0.7 .* yh), eh, color=CTOT, whiskerwidth=8, linewidth=1.3)
scatter!(ax2, xh, yh, color=CTOT, markersize=13, strokecolor=:white, strokewidth=1,
         label="hse ladder")
scatter!(ax2, [s[1] for s in SC14_HI], [s[5]*1e-2 for s in SC14_HI], color=:white,
         strokecolor=(:black, 0.6), strokewidth=1.3, markersize=9, marker=:circle,
         label="SC14 hi-res")
scatter!(ax2, [s[1] for s in SC14_STD], [s[5]*1e-2 for s in SC14_STD], color=:white,
         strokecolor=(:black, 0.55), strokewidth=1.4, markersize=11, marker=:rect,
         label="SC14 std-res")
axislegend(ax2, position=:lb, framevisible=false, labelsize=11, rowgap=0)

Label(fig[2, 1:2],
      "Filled: hse_outflow ladder (β = 10 from scratch, then each β restarted from its parent's end).  Open, dashed: the same ladder on the diode boundary " *
      "(its β = 80 averaged over the first 200/Ω, before its z-boundary mass runaway).\n" *
      "Trailing 100/Ω windows (200/Ω at β ≥ 40); history file at 0.25/Ω; error bars = s.e.m. with a 10/Ω correlation time.   □ SC14 standard res, ○ SC14 hi-res (Table 1).",
      fontsize=11, color=:gray30, justification=:left, lineheight=1.25, padding=(0, 0, 2, 6))
save(joinpath(@__DIR__, "sc14_alpha_vs_beta_hse.png"), fig; px_per_unit=2)
println("ok")
