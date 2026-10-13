# Stress vs cooling time for the MORPHED standard-resolution beta chain
# (256 x 256 x 48, box 64H x 64H x 12H, nghost = 4), in the form of Shi & Chiang
# (2014) Figures 3 and 4.
#
# Unlike sc14_alpha_vs_beta.png -- whose beta != 10 runs each started from the cold
# Q0 = 1 initial condition -- these follow SC14's own morphing ladder, each stage
# initialized from the final state of the previous one:
#
#     beta = 10 anchor (t = 0-300, from scratch)
#        |-> beta =  4   (t = 300-500)
#        |-> beta = 20   (t = 300-450)
#                          |-> beta = 40   (t = 450-750)
#                                            |-> beta = 80   (t = 750-1150)
#
# The lineage is not guesswork: each run's output file numbering continues its
# parent's (b4/b20 start at index 31 after the anchor's 0-30; b40 starts at 46
# after b20's 45), so beta = 40 descends from beta = 20, not from the anchor.
#
#   left  (their Fig. 3): density-weighted alpha  = <w_xy>_rho / <P>_rho   (eq. 19)
#   right (their Fig. 4): volume-averaged  alpha' = (2/3) <w_xy> / <gamma P> (eq. 20)
#                         against the parameter-free eq. 21.
#
# Filled circles are from the history file (0.25/Omega); crosses are the same
# quantity recomputed from the .bin dumps by scripts/extract_gt_stress.py (w_xy
# summed per cell, then integrated).  The two are identical at matched times, but
# the dumps are 10/Omega apart -- only 11-21 samples per averaging window against
# the history's 401-801 -- and the instantaneous Reynolds stress has a scatter
# comparable to its own mean, so the dump estimate of that channel goes NEGATIVE at
# beta = 20 and 40 here.  The history is therefore the primary series; the earlier
# from-scratch figure (sc14_alpha_vs_beta.png) had the roles the other way round
# because that figure was specifically about the .bin recomputation.
# Open grey diamonds are the corresponding FRESH-START runs, where they exist, so
# the effect of morphing is visible.
#
# Usage: julia validation/figures/plot_sc14_alpha_vs_beta_morph.jl
using CairoMakie, Printf, DelimitedFiles, Statistics, PyCall
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))

const HERE  = @__DIR__
const RUN   = joinpath(dirname(HERE), "run")
const GAMMA = 5/3
const TCORR = 10.0

# beta, run directory, trailing averaging window (SC14 Table 1 convention)
# explicit (t_lo, t_hi) windows: trailing 100/Omega (200/Omega at beta >= 40) as in
# SC14 Table 1, except beta = 80 which uses its FIRST 200/Omega -- see above
const RUNS = [(4.0,  "gt_sc14_stndrd_b4_morph",     (400.0,  500.0)),
              (10.0, "gt_sc14_stndrd_b10_ng4",      (200.0,  300.0)),
              (20.0, "gt_sc14_stndrd_b20_morph",    (350.0,  450.0)),
              (40.0, "gt_sc14_stndrd_b40_morph",    (550.0,  750.0)),
              (80.0, "gt_sc14_stndrd_b80_morphing", (750.0,  950.0))]
# beta = 80 needs an EARLY averaging window.  Morphing delays the z-boundary mass
# runaway that destroyed the from-scratch beta = 80, but does not prevent it: dM/dt
# turns sustainedly positive ~63/Omega in, and by t = 1150 the box has gained 97% of
# its mass with <cs>_rho at 8.1.  SC14's trailing 200/Omega (t = 950-1150) therefore
# measures the runaway, not the disk -- alpha' there is 0.13e-2, a quarter of eq. 21.
# The first 200/Omega (t = 750-950, mean M/M0 = 1.014) is still quasi-steady and is
# what the plotted point uses; the contaminated trailing value is shown as a grey
# cross with a downward arrow so the size of the effect is visible.
const B80_LATE = (750.0, 950.0, 950.0, 1150.0)   # (clean lo, clean hi, bad lo, bad hi)

# beta = 3 with AMR, also morphed from the anchor at t = 300, but it fragments: no
# steady state, so it is drawn as a star from its whole 25/Omega run and kept off
# the trend line (SC14 tabulate no averages for their tc=3 either)
const B3 = (3.0, "gt_sc14_stndrd_b3_amr")
# the from-scratch counterparts, for the morphing comparison (no beta = 20 exists)
# (the from-scratch beta = 80 is deliberately absent: it suffered the same z-boundary
#  runaway far worse -- M x 9 -- so it is not a measurement of anything)
const FRESH = [(4.0,  "gt_sc14_stndrd_b4",  (100.0, 200.0)),
               (10.0, "gt_sc14_stndrd_b10", (200.0, 300.0)),
               (40.0, "gt_sc14_stndrd_b40", (100.0, 300.0))]

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
    keep = falses(size(d,1)); tmin = Inf
    for i in size(d,1):-1:1
        d[i,1] < tmin && (keep[i] = true; tmin = d[i,1])
    end
    d = d[keep, :]
    (t=d[:,1], g=d[:,5]./d[:,7], r=d[:,6]./d[:,7],
     ab=(2/3).*(d[:,8].+d[:,9])./d[:,10])
end

function read_stress(dir)
    f = joinpath(RUN, dir, "gt_stress.npz")
    isfile(f) || return nothing
    d = pyimport("numpy").load(f)
    g(k) = convert(Vector{Float64}, get(d, k))
    (t=g("time"), g=g("rho_wgrv")./g("rho_prs"), r=g("rho_wrey")./g("rho_prs"),
     tot=g("rho_wtot")./g("rho_prs"), ab=(2/3).*g("wtot")./g("rho_cs2"))
end

wmean(t, y, w) = (m = (t .>= w[1]) .& (t .<= w[2]);
                  count(m) == 0 ? (NaN, NaN, 0) :
                  (mean(y[m]), std(y[m])/sqrt(max(1,count(m))), count(m)))
# the history samples at 0.25/Omega, far finer than the ~10/Omega correlation time,
# so its s.e.m. uses T/TCORR independent samples rather than the raw row count
whist(t, y, w) = (m = (t .>= w[1]) .& (t .<= w[2]);
                  (mean(y[m]), std(y[m])/sqrt(max(1.0, (w[2]-w[1])/TCORR))))

B = Float64[]; bg=Float64[]; bge=Float64[]; br=Float64[]; bre=Float64[]
bt=Float64[]; bte=Float64[]; bb=Float64[]; bbe=Float64[]; nb=Int[]
hg=Float64[]; hr=Float64[]; ht=Float64[]; hb=Float64[]
hge=Float64[]; hre=Float64[]; hte=Float64[]; hbe=Float64[]
@printf("%-5s %-12s %4s | %7s %7s %8s %8s | %8s\n",
        "beta","window","N","aRey","aGrav","a(19)","a'(20)","eq21")
for (beta, dir, w) in RUNS
    s = read_stress(dir); h = read_hist(dir)
    # beta = 80 has no grav_phi dumps, so there is no .bin estimate for it
    if s === nothing
        (g,ge,r,re,tt,te,ab,abe,n) = (NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0)
    else
        (g,ge,n) = wmean(s.t, s.g, w); (r,re,_) = wmean(s.t, s.r, w)
        (tt,te,_) = wmean(s.t, s.tot, w); (ab,abe,_) = wmean(s.t, s.ab, w)
    end
    push!(B,beta); push!(bg,g); push!(bge,ge); push!(br,r); push!(bre,re)
    push!(bt,tt); push!(bte,te); push!(bb,ab); push!(bbe,abe); push!(nb,n)
    (v,e) = whist(h.t, h.g, w);            push!(hg,v); push!(hge,e)
    (v,e) = whist(h.t, h.r, w);            push!(hr,v); push!(hre,e)
    (v,e) = whist(h.t, h.g.+h.r, w);       push!(ht,v); push!(hte,e)
    (v,e) = whist(h.t, h.ab, w);           push!(hb,v); push!(hbe,e)
    @printf("%-5.0f %4.0f-%-7.0f %4d | %7.2f %7.2f %8.2f %8.2f | %8.2f\n",
            beta, w[1], w[2], n, 100r, 100g, 100tt, 100ab,
            100*4/(9*GAMMA*(GAMMA-1))/beta)
end

# beta = 3: whole-run means (history primary, .bin cross), error bar over 25/Omega
h3 = read_hist(B3[2]); s3 = read_stress(B3[2]); w3 = (h3.t[1], h3.t[end])
(h3t, h3te) = whist(h3.t, h3.g .+ h3.r, w3); (h3b, h3be) = whist(h3.t, h3.ab, w3)
b3t = mean(s3.tot); b3b = mean(s3.ab)
@printf("%-5.0f %4.0f-%-7.0f %4d | %7.2f %7.2f %8.2f %8.2f | %8.2f   (whole run, fragments)\n",
        3.0, h3.t[1], h3.t[end], length(s3.t), 100*mean(h3.r), 100*mean(h3.g), 100h3t, 100h3b,
        100*4/(9*GAMMA*(GAMMA-1))/3.0)

# beta = 80 in SC14's own trailing window, i.e. inside the runaway
h80 = read_hist("gt_sc14_stndrd_b80_morphing")
(b80t, _) = whist(h80.t, h80.g .+ h80.r, (B80_LATE[3], B80_LATE[4]))
(b80b, _) = whist(h80.t, h80.ab,         (B80_LATE[3], B80_LATE[4]))
@printf("%-5.0f %4.0f-%-7.0f %4s | %7s %7s %8.2f %8.2f |   (SC14 trailing window: inside the mass runaway)\n",
        80.0, B80_LATE[3], B80_LATE[4], "-", "-", "-", 100b80t, 100b80b)

# fresh-start counterparts (history file only; that is the better time average)
fB = Float64[]; ft = Float64[]; fb = Float64[]
for (beta, dir, w) in FRESH
    h = read_hist(dir); m = (h.t .>= w[1]) .& (h.t .<= w[2])
    push!(fB, beta); push!(ft, mean(h.g[m].+h.r[m])); push!(fb, mean(h.ab[m]))
end

const CTOT = :black
const CGRV = RGBf(0.80, 0.14, 0.22)
const CREY = RGBf(0.17, 0.42, 0.72)
const YFLOOR = 1.05e-3
ok(v)   = !isnan(v) && v > 0
neg(v)  = !isnan(v) && v <= 0            # measured and non-positive: park on the floor
gap(y)  = [ok(v) ? v : NaN for v in y]   # NaN breaks the polyline at gaps
possplit(x,y,e) = (x[ok.(y)], y[ok.(y)], e[ok.(y)], x[neg.(y)])

fig = Figure(size=(1180, 570))
Label(fig[0, 1:2],
      "SC14 morphed β-ladder at standard resolution (256×256×48, nghost = 4): stress vs cooling time",
      fontsize=18, font=:bold, padding=(0,0,4,0))

betagrid = 10 .^ range(log10(2.4), log10(140), length=200)
function panel(gp; ylab, title)
    ax = Axis(gp; xscale=log10, yscale=log10, xlabel="Ω t_cool  ( = β )",
              ylabel=ylab, title=title, titlealign=:left, titlefont=:bold,
              xticks=([3,4,5,10,20,40,80,120], ["3","4","5","10","20","40","80","120"]),
              yticks=([1e-3,3e-3,1e-2,3e-2,1e-1,3e-1],
                      ["10⁻³","3×10⁻³","10⁻²","3×10⁻²","10⁻¹","3×10⁻¹"]),
              xminorticksvisible=true, yminorticksvisible=true,
              xgridcolor=(:gray,0.18), ygridcolor=(:gray,0.18))
    xlims!(ax, 2.4, 140); ylims!(ax, 8e-4, 4e-1)
    return ax
end

# ------------------------------------------------------------------ panel (a)
ax1 = panel(fig[1,1]; ylab="α  =  ⟨w_xy⟩_ρ / ⟨P⟩_ρ    (SC14 eq. 19)",
            title="(a)  density-weighted stress — SC14 Fig. 3")
lines!(ax1, betagrid, 0.576 ./ betagrid, color=(:gray,0.75), linestyle=:dash,
       linewidth=2, label="∝ 1/(Ω t_cool)")
for (y,e,yb,col,lab) in ((ht,hte,bt,CTOT,"total"), (hg,hge,bg,CGRV,"gravitational"),
                         (hr,hre,br,CREY,"Reynolds"))
    x,yy,ee,xneg = possplit(B,y,e)
    lines!(ax1, B, gap(y), color=col, linewidth=2.2)      # breaks at negative points
    errorbars!(ax1, x, yy, min.(ee, 0.7.*yy), ee, color=col, whiskerwidth=8, linewidth=1.4)
    scatter!(ax1, x, yy, color=col, markersize=13, strokecolor=:white, strokewidth=1,
             label="morphed, "*lab)
    isempty(xneg) || scatter!(ax1, xneg, fill(YFLOOR,length(xneg)), color=(col,0.35),
                              strokecolor=col, strokewidth=1.6, marker=:dtriangle, markersize=13)
    scatter!(ax1, B[ok.(yb)], yb[ok.(yb)], color=col, marker=:xcross, markersize=11)
    xn = B[neg.(yb)]                              # measured negative -> floor; NaN -> nothing
    isempty(xn) || scatter!(ax1, xn, fill(YFLOOR,length(xn)), color=col, marker=:xcross,
                            markersize=11)
end
for (col,idx) in ((CTOT,4),(CGRV,3),(CREY,2))
    x=[s[1] for s in SC14_HI]; y=[s[idx+1]*1e-2 for s in SC14_HI]
    lines!(ax1, x, y, color=(col,0.45), linestyle=:dot, linewidth=1.6)
    scatter!(ax1, x, y, color=:white, strokecolor=col, strokewidth=1.6, markersize=10)
end
scatter!(ax1, [s[1] for s in SC14_STD], [s[4]*1e-2 for s in SC14_STD], color=:white,
         strokecolor=(:black,0.55), strokewidth=1.4, markersize=11, marker=:rect)
scatter!(ax1, fB, ft, color=(:gray,0.15), strokecolor=(:gray,0.85), strokewidth=1.5,
         marker=:diamond, markersize=13, label="from scratch (total)")
text!(ax1, 28.0, YFLOOR*1.15; text="✕ .bin estimate negative", align=(:center,:bottom),
      color=CREY, fontsize=10.5)
const CB80 = RGBf(0.45, 0.45, 0.45)
scatter!(ax1, [80.0], [b80t], color=CB80, marker=:xcross, markersize=11)
text!(ax1, 80.0, b80t/1.4; text="β = 80: ● uses the FIRST 200/Ω\n✕ = SC14's trailing window,\n      inside the mass runaway",
      align=(:center,:top), color=CB80, fontsize=9.5)
const CB3 = RGBf(0.55, 0.25, 0.65)
errorbars!(ax1, [3.0], [h3t], [min(h3te, 0.7h3t)], [h3te], color=CB3, whiskerwidth=8, linewidth=1.4)
scatter!(ax1, [3.0], [h3t], color=(CB3,0.35), strokecolor=CB3, strokewidth=2, marker=:star5,
         markersize=18, label="β = 3, AMR: fragments (whole 25/Ω run)")
scatter!(ax1, [3.0], [b3t], color=CB3, marker=:xcross, markersize=11)
axislegend(ax1, position=:lb, framevisible=false, labelsize=11, rowgap=0)

# ------------------------------------------------------------------ panel (b)
ax2 = panel(fig[1,2]; ylab="α′ = (2/3)⟨w_xy⟩ / ⟨γP⟩    (SC14 eq. 20)",
            title="(b)  volume-averaged stress — SC14 Fig. 4")
lines!(ax2, betagrid, (4/(9*GAMMA*(GAMMA-1))) ./ betagrid, color=:black,
       linestyle=:dash, linewidth=2, label="eq. 21:  4 / [9γ(γ−1) Ω t_cool]")
lines!(ax2, B, hb, color=CTOT, linewidth=2.2)
errorbars!(ax2, B, hb, min.(hbe,0.7.*hb), hbe, color=CTOT, whiskerwidth=8, linewidth=1.4)
scatter!(ax2, B, hb, color=CTOT, markersize=13, strokecolor=:white, strokewidth=1,
         label="morphed, from history file")
scatter!(ax2, B[ok.(bb)], bb[ok.(bb)], color=CTOT, marker=:xcross, markersize=11,
         label="morphed, from .bin dumps")
xh=[s[1] for s in SC14_HI]; yh=[s[5]*1e-2 for s in SC14_HI]
lines!(ax2, xh, yh, color=(:black,0.45), linestyle=:dot, linewidth=1.6)
scatter!(ax2, xh, yh, color=:white, strokecolor=(:black,0.8), strokewidth=1.6,
         markersize=10, label="SC14 hi-res (Table 1)")
scatter!(ax2, [s[1] for s in SC14_STD], [s[5]*1e-2 for s in SC14_STD], color=:white,
         strokecolor=(:black,0.55), strokewidth=1.4, markersize=11, marker=:rect,
         label="SC14 std-res (Table 1)")
scatter!(ax2, fB, fb, color=(:gray,0.15), strokecolor=(:gray,0.85), strokewidth=1.5,
         marker=:diamond, markersize=13, label="from scratch")
scatter!(ax2, [80.0], [b80b], color=CB80, marker=:xcross, markersize=11)
text!(ax2, 62.0, b80b; text="β = 80: ● first 200/Ω,\n✕ SC14's trailing window",
      align=(:right,:center), color=CB80, fontsize=9.5)
errorbars!(ax2, [3.0], [h3b], [min(h3be, 0.7h3b)], [h3be], color=CB3, whiskerwidth=8, linewidth=1.4)
scatter!(ax2, [3.0], [h3b], color=(CB3,0.35), strokecolor=CB3, strokewidth=2, marker=:star5,
         markersize=18, label="β = 3, AMR: fragments (whole 25/Ω run)")
scatter!(ax2, [3.0], [b3b], color=CB3, marker=:xcross, markersize=11)
axislegend(ax2, position=:lb, framevisible=false, labelsize=11, rowgap=0)

Label(fig[2, 1:2],
      "Ladder: β = 10 anchor (from scratch, Ωt = 0–300) → β = 3 (AMR), β = 4 and β = 20 → β = 40 → β = 80, each stage restarted from its parent's final state.  Trailing 100/Ω\n" *
      "averages (200/Ω at β ≥ 40), except β = 80, which uses its FIRST 200/Ω: the z-boundary mass runaway that wrecked the from-scratch β = 80 is only delayed by morphing, and\n" *
      "SC14's trailing window falls inside it (grey ✕).  β = 3 fragments, so its star is the whole-run mean, off the trend and with an error bar over only ~2.5 correlation times.\n" *
      "● from the history file at 0.25/Ω (error bar = s.e.m. with a 10/Ω correlation time)   ✕ same quantity from the 10/Ω .bin dumps — too few samples for the Reynolds channel,\n" *
      "which its estimate drives negative at β = 20 and 40 (parked on the axis floor); β = 80 has no grav_phi dumps, so it has no ✕ for the components.  Lines break where a value\n" *
      "is non-positive.   ◇ the from-scratch runs of the previous figure   ○ SC14 hi-res, □ SC14 standard-res (Table 1)",
      fontsize=11, color=:gray30, justification=:left, lineheight=1.25, padding=(0,0,2,6))
rowgap!(fig.layout, 4)

out = joinpath(HERE, "sc14_alpha_vs_beta_morph.png")
save(out, fig; px_per_unit=2)
println("wrote $out")
