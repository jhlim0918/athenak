# Stress vs cooling time for the standard-resolution SC14 beta scan
# (256 x 256 x 48, box 64H x 64H x 12H), in the form of Shi & Chiang (2014)
# Figures 3 and 4.
#
#   left  (their Fig. 3): density-weighted  alpha  = <w_xy>_rho / <P>_rho   (eq. 19)
#   right (their Fig. 4): volume-averaged   alpha' = (2/3) <w_xy> / <gamma P>  (eq. 20)
#                         against the parameter-free prediction of eq. 21,
#                         alpha' = 4/(9 gamma (gamma-1)) / (Omega t_cool).
#
# Filled circles are recomputed from the .bin dumps by scripts/extract_gt_stress.py,
# which forms w_xy = g_x g_y/4piG + rho v_x dv_y PER CELL from the hydro_w/grav_phi
# pair and only then integrates.  Small crosses are the same quantity from the
# in-code history file.  The two agree to 6 digits snapshot by snapshot (the
# domain integral is linear, so summing the two stress channels before or after
# integrating is identical); they differ only in time sampling -- the dumps are
# spaced 10/Omega (11-30 samples per window) against the history's 0.25/Omega
# (401-801), and the instantaneous stress scatter is comparable to its own mean,
# so the history remains the better time average.  Both are plotted for that reason.
#
# Points are time averages over the trailing window of each run (SC14 Table 1
# convention: 100/Omega for beta <= 20, 200/Omega for beta >= 40); error bars are
# the standard error of the mean over the actual number of samples.
# Open symbols in matching colors are SC14's own high-resolution Table-1 values.
#
# Runs flagged non-steady are drawn but excluded from the trend:
#   beta = 3  ran only 30/Omega after morphing out of beta = 10 and is
#             burst-dominated (SC14 likewise tabulate no averages for tc=3.hi,
#             which fragmented).  No Truelove-violating clump has formed yet at
#             this resolution: rho_max = 11 rho_0, Jeans length ~ 13 cells.
#   beta = 80 suffered a z-boundary mass/energy runaway (M x 9, E x 870) -- the
#             from-scratch start left the disk too hot for beta=80 cooling to
#             recover, unlike SC14 who morphed 80 out of 40.
#
# Usage: julia validation/figures/plot_sc14_alpha_vs_beta.jl
using CairoMakie, Printf, DelimitedFiles, Statistics, PyCall
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))

const HERE   = @__DIR__
const RUN    = joinpath(dirname(HERE), "run")
const GAMMA  = 5/3
const TCORR  = 10.0          # turbulence correlation time, for the history error bars

# beta, run directory, averaging window, status
const RUNS = [
    (3.0,  "gt_sc14_stndrd_b3_morphing", 30.0,  :hot),
    (4.0,  "gt_sc14_stndrd_b4",         100.0,  :ok),
    (5.0,  "gt_sc14_stndrd_b5",         100.0,  :ok),
    (10.0, "gt_sc14_stndrd_b10",        100.0,  :ok),
    (40.0, "gt_sc14_stndrd_b40",        200.0,  :ok),
    (80.0, "gt_sc14_stndrd_b80",        200.0,  :bad),
]

# SC14 Table 1, high-resolution constant-beta runs (the data behind their Figs. 3-4):
#   beta, alpha_Reynolds, alpha_gravity, alpha (eq. 19), alpha' (eq. 20), all x 1e-2
const SC14_HI = [
    (4.0,   5.69, 6.50, 12.2, 10.0),
    (5.0,   4.19, 5.82, 10.0,  8.19),
    (8.0,   2.06, 4.41,  6.47, 5.10),
    (10.0,  1.70, 3.82,  5.52, 4.06),
    (20.0,  0.46, 2.42,  2.88, 2.02),
    (40.0,  0.27, 1.36,  1.62, 1.03),
    (80.0,  0.35, 0.58,  0.93, 0.52),
    (120.0, 0.14, 0.39,  0.53, 0.31),
]
# SC14 Table 1, standard-resolution runs -- the direct counterparts of ours
const SC14_STD = [
    (10.0, 1.88, 3.87, 5.76, 4.24),
    (20.0, 0.29, 2.55, 2.85, 2.14),
    (40.0, -0.06, 1.51, 1.44, 1.06),
    (80.0, 0.13, 0.76, 0.89, 0.55),
]

"Read a run's user history, dropping the rewound rows of restart seams."
function read_hist(dir)
    files = filter(f -> endswith(f, ".user.hst"), readdir(joinpath(RUN, dir), join=true))
    isempty(files) && error("no *.user.hst in $dir")
    d = readdlm(files[1], comments=true, comment_char='#')
    keep = falses(size(d, 1)); tmin = Inf
    for i in size(d, 1):-1:1
        d[i, 1] < tmin && (keep[i] = true; tmin = d[i, 1])
    end
    return d[keep, :]
end

"Window mean and standard error of the mean of y over the trailing Tavg."
function wavg(t, y, Tavg)
    m = t .>= (t[end] - Tavg)
    yy = y[m]
    nind = max(1.0, (t[end] - t[m][1])/TCORR)
    return mean(yy), std(yy)/sqrt(nind)
end

"Window mean and s.e.m. of y over the trailing Tavg, treating every sample as
 independent (the .bin dumps are spaced >= one correlation time apart)."
function wavg_indep(t, y, Tavg)
    m = t .>= (t[end] - Tavg)
    yy = y[m]
    return mean(yy), std(yy)/sqrt(max(1, length(yy))), length(yy)
end

"Stress integrals recomputed from the .bin dumps (scripts/extract_gt_stress.py)."
function read_stress(dir)
    f = joinpath(RUN, dir, "gt_stress.npz")
    isfile(f) || return nothing
    d = pyimport("numpy").load(f)
    g(k) = convert(Vector{Float64}, get(d, k))
    return (t=g("time"), g=g("rho_wgrv")./g("rho_prs"), r=g("rho_wrey")./g("rho_prs"),
            tot=g("rho_wtot")./g("rho_prs"), ab=(2/3).*g("wtot")./g("rho_cs2"))
end

rows = NamedTuple[]
brows = Dict{Float64,NamedTuple}()
for (beta, dir, Tavg, status) in RUNS
    d = read_hist(dir)
    t = d[:, 1]
    ag = d[:, 5] ./ d[:, 7]                        # rho_wgrv / rho_prs
    ar = d[:, 6] ./ d[:, 7]                        # rho_wrey / rho_prs
    ab = (2/3) .* (d[:, 8] .+ d[:, 9]) ./ d[:, 10] # (2/3)(wgrv+wrey)/(gamma int P)
    Tavg = min(Tavg, t[end] - t[1])
    g, ge = wavg(t, ag, Tavg); r, re = wavg(t, ar, Tavg)
    tot, te = wavg(t, ag .+ ar, Tavg); b, be = wavg(t, ab, Tavg)
    push!(rows, (beta=beta, status=status, t0=t[end]-Tavg, t1=t[end],
                 g=g, ge=ge, r=r, re=re, tot=tot, te=te, ab=b, abe=be))

    s = read_stress(dir)
    if s !== nothing
        Tb = min(Tavg, s.t[end] - s.t[1])
        bg, bge, n  = wavg_indep(s.t, s.g,   Tb)
        br, bre, _  = wavg_indep(s.t, s.r,   Tb)
        bt, bte, _  = wavg_indep(s.t, s.tot, Tb)
        bb, bbe, _  = wavg_indep(s.t, s.ab,  Tb)
        brows[beta] = (beta=beta, status=status, n=n, g=bg, ge=bge, r=br, re=bre,
                       tot=bt, te=bte, ab=bb, abe=bbe)
    end
end

@printf("%-5s %-5s %-12s %-4s | %-25s | %-25s | %6s\n", "beta", "state", "window",
        "Nbin", "  from .bin dumps (x1e-2)", "  from history   (x1e-2)", "eq21")
@printf("%-5s %-5s %-12s %-4s | %7s %7s %8s | %7s %7s %8s | %6s\n", "", "", "", "",
        "aRey", "aGrav", "a'(20)", "aRey", "aGrav", "a'(20)", "")
for w in rows
    b = get(brows, w.beta, nothing)
    bs = b === nothing ? "      -       -        -" :
         @sprintf("%7.2f %7.2f %8.2f", 100b.r, 100b.g, 100b.ab)
    @printf("%-5.0f %-5s %4.0f-%-7.0f %-4s | %s | %7.2f %7.2f %8.2f | %6.2f\n",
            w.beta, w.status, w.t0, w.t1,
            b === nothing ? "-" : string(b.n), bs,
            100w.r, 100w.g, 100w.ab, 100*4/(9*GAMMA*(GAMMA-1))/w.beta)
end

ok(w)   = w.status == :ok
okbetas = [w.beta for w in rows if ok(w)]
bsel(f)    = (okbetas, [f(brows[b]) for b in okbetas])
bselerr(f) = [f(brows[b]) for b in okbetas]
hsel(f)    = (okbetas, [f(w) for w in rows if ok(w)])

# log axes cannot show a non-positive stress: split those off and park them on the
# floor with a down-triangle.  A Reynolds stress consistent with zero at long
# cooling times is a real feature -- SC14's own standard-res tc=40 gives -0.06e-2.
const YFLOOR = 1.05e-3
possplit(x, y, e) = (x[y .> 0], y[y .> 0], e[y .> 0], x[y .<= 0])

const CTOT = :black
const CGRV = RGBf(0.80, 0.14, 0.22)
const CREY = RGBf(0.17, 0.42, 0.72)

fig = Figure(size=(1180, 560))
Label(fig[0, 1:2],
      "SC14 β-scan at standard resolution (256×256×48, 64H×64H×12H): stress vs cooling time",
      fontsize=18, font=:bold, padding=(0, 0, 4, 0))

betagrid = 10 .^ range(log10(2.4), log10(140), length=200)

function panel(gp; ylab, title)
    ax = Axis(gp; xscale=log10, yscale=log10, xlabel="Ω t_cool  ( = β )",
              ylabel=ylab, title=title, titlealign=:left, titlefont=:bold,
              xticks=([3, 5, 10, 20, 40, 80, 120], ["3","5","10","20","40","80","120"]),
              yticks=([1e-3,3e-3,1e-2,3e-2,1e-1,3e-1],
                      ["10⁻³","3×10⁻³","10⁻²","3×10⁻²","10⁻¹","3×10⁻¹"]),
              xminorticksvisible=true, yminorticksvisible=true,
              xgridcolor=(:gray, 0.18), ygridcolor=(:gray, 0.18))
    xlims!(ax, 2.4, 140); ylims!(ax, 8e-4, 6e-1)
    return ax
end

# ---------------------------------------------------------------- panel (a)
ax1 = panel(fig[1, 1]; ylab="α  =  ⟨w_xy⟩_ρ / ⟨P⟩_ρ    (SC14 eq. 19)",
            title="(a)  density-weighted stress — SC14 Fig. 3")

lines!(ax1, betagrid, 0.576 ./ betagrid, color=(:gray, 0.75), linestyle=:dash,
       linewidth=2, label="∝ 1/(Ω t_cool)")

for (f, ferr, col, lab) in ((w -> w.tot, w -> w.te, CTOT, "total"),
                            (w -> w.g,   w -> w.ge, CGRV, "gravitational"),
                            (w -> w.r,   w -> w.re, CREY, "Reynolds"))
    x0, y0 = bsel(f); e0 = bselerr(ferr)
    x, y, e, xneg = possplit(x0, y0, e0)
    lines!(ax1, x, y, color=col, linewidth=2.2)
    errorbars!(ax1, x, y, min.(e, 0.7 .* y), e, color=col, whiskerwidth=8, linewidth=1.4)
    scatter!(ax1, x, y, color=col, markersize=13, strokecolor=:white, strokewidth=1,
             label="from .bin, " * lab)
    isempty(xneg) || scatter!(ax1, xneg, fill(YFLOOR, length(xneg)), color=(col, 0.35),
                              strokecolor=col, strokewidth=1.6, marker=:dtriangle,
                              markersize=13)
    xh0, yh0 = hsel(f)
    ph = yh0 .> 0
    scatter!(ax1, xh0[ph], yh0[ph], color=col, marker=:xcross, markersize=11)
end

for (col, idx) in ((CTOT, 4), (CGRV, 3), (CREY, 2))
    x = [s[1] for s in SC14_HI]; y = [s[idx]*1e-2 for s in SC14_HI]
    lines!(ax1, x, y, color=(col, 0.45), linestyle=:dot, linewidth=1.6)
    scatter!(ax1, x, y, color=:white, strokecolor=col, strokewidth=1.6, markersize=10,
             marker=:circle)
end
x = [s[1] for s in SC14_STD]; y = [s[4]*1e-2 for s in SC14_STD]
scatter!(ax1, x, y, color=:white, strokecolor=(:black, 0.55), strokewidth=1.4,
         markersize=11, marker=:rect)

# ---------------------------------------------------------------- panel (b)
ax2 = panel(fig[1, 2]; ylab="α′ = (2/3)⟨w_xy⟩ / ⟨γP⟩    (SC14 eq. 20)",
            title="(b)  volume-averaged stress — SC14 Fig. 4")

lines!(ax2, betagrid, (4/(9*GAMMA*(GAMMA-1))) ./ betagrid, color=:black,
       linestyle=:dash, linewidth=2,
       label="eq. 21:  4 / [9γ(γ−1) Ω t_cool]")
x, y, e, _ = possplit(bsel(w -> w.ab)..., bselerr(w -> w.abe))
lines!(ax2, x, y, color=CTOT, linewidth=2.2)
errorbars!(ax2, x, y, min.(e, 0.7 .* y), e, color=CTOT, whiskerwidth=8, linewidth=1.4)
scatter!(ax2, x, y, color=CTOT, markersize=13, strokecolor=:white, strokewidth=1,
         label="from .bin dumps")
xh, yh = hsel(w -> w.ab)
scatter!(ax2, xh, yh, color=CTOT, marker=:xcross, markersize=11,
         label="from history file")
xh = [s[1] for s in SC14_HI]; yh = [s[5]*1e-2 for s in SC14_HI]
lines!(ax2, xh, yh, color=(:black, 0.45), linestyle=:dot, linewidth=1.6)
scatter!(ax2, xh, yh, color=:white, strokecolor=(:black, 0.8), strokewidth=1.6,
         markersize=10, label="SC14 hi-res (Table 1)")
xs = [s[1] for s in SC14_STD]; ys = [s[5]*1e-2 for s in SC14_STD]
scatter!(ax2, xs, ys, color=:white, strokecolor=(:black, 0.55), strokewidth=1.4,
         markersize=11, marker=:rect, label="SC14 std-res (Table 1)")

# ------------------------------------------------- non-steady runs, both panels
for (ax, f) in ((ax1, w -> w.tot), (ax2, w -> w.ab))
    for w in rows
        w.status == :ok && continue
        col = w.status == :hot ? RGBf(0.55, 0.25, 0.65) : RGBf(0.45, 0.45, 0.45)
        v = haskey(brows, w.beta) ? f(brows[w.beta]) : f(w)
        scatter!(ax, [w.beta], [v], color=(col, 0.35), strokecolor=col,
                 strokewidth=2, markersize=17,
                 marker=(w.status == :hot ? :star5 : :xcross))
    end
end
text!(ax1, 3.25, 0.185; text="β = 3: 30/Ω only,\nno steady state", align=(:left, :bottom),
      color=RGBf(0.55, 0.25, 0.65), fontsize=11.5)
text!(ax1, 80.0, 1.7e-3; text="β = 80\nboundary runaway", align=(:center, :bottom),
      color=RGBf(0.4, 0.4, 0.4), fontsize=11.5)
text!(ax1, 40.0, YFLOOR*1.1; text="▽ negative", align=(:center, :bottom),
      color=CREY, fontsize=10.5)
text!(ax2, 3.25, 0.185; text="β = 3: 30/Ω only,\nno steady state", align=(:left, :bottom),
      color=RGBf(0.55, 0.25, 0.65), fontsize=11.5)
text!(ax2, 80.0, 1.35e-3; text="β = 80\nboundary runaway", align=(:center, :bottom),
      color=RGBf(0.4, 0.4, 0.4), fontsize=11.5)

axislegend(ax1, position=:lb, framevisible=false, labelsize=11.5, rowgap=0)
axislegend(ax2, position=:lb, framevisible=false, labelsize=11.5, rowgap=0)
Label(fig[2, 1:2],
      "● recomputed from the .bin dumps: w_xy summed per cell, then integrated (error bar = s.e.m. over the 11–21 dumps in the window)\n" *
      "✕ same quantity from the history file — identical at matched times, but 0.25/Ω sampling, so a far better time average\n" *
      "○ SC14 hi-res, □ SC14 standard-res (Table 1)     ▽ estimate is negative, parked on the axis floor",
      fontsize=11.5, color=:gray30, justification=:left, lineheight=1.25,
      padding=(0, 0, 2, 6))
rowgap!(fig.layout, 4)

out = joinpath(HERE, "sc14_alpha_vs_beta.png")
save(out, fig; px_per_unit=2)
println("wrote $out")
