# alpha' against the cooling time beta in SC14 gravito-turbulence: does the stress balance the
# beta cooling (SC14 eq. 21, alpha' = 4/[9 gamma (gamma-1) beta]) at every beta?  2026-11-05.
# (a) alpha' (SC14 eq. 20, (2/3)(w_grav + w_Rey)/<rho cs^2>) against beta with eq. 21;
# (b) alpha'/eq.21; (c) the Reynolds share of alpha'; (d) the energy budget of the WENO-Z
#     ladder (scripts/analysis/gt_energy_budget.py: stress work S, energy leaving through the
#     z faces F_z, and the unattributed residual R, as fractions of the cooling C), with the
#     PPM4 ladder at beta = 40 and 80 and the heating plm lacks at beta = 80 (1 - S/C).
# Runs:
#   WENO-Z + FOFC ladder, production box (64H x 64H x 16H, 4 cells/H, mass renormalization):
#     validation/run/gt_l64z16/{b4lad,b5lad,b20lad,b40lad,b80lad}_renorm_r4 (morphs of the beta
#     ramp run b10ramp_renorm_r4 at t = 100; beta = 4, 5 directly, 20 -> 40 -> 80 in sequence)
#     and b10ramp_renorm_r4 itself (t = 75-100, after the ramp);
#   plm and PPM4 ladders and the WENO-Z+FOFC beta = 10 run, SC14 standard box (no mass
#     renormalization): validation/run/gt_sc14_stndrd_b{10,20,40,80}_hse{,_ppm4,_wzfofc}.
# alpha' is averaged over each stage's final window; error bars: standard error of the 5/Omega
# block means (blocks are correlated, so approximate).
# Usage: julia validation/figures/plot_alpha_beta.jl
using CairoMakie, DelimitedFiles, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
RUN = joinpath(dirname(@__DIR__), "run")
gam = 5/3
eq21(b) = 4/(9gam*(gam - 1)*b)

function hst(f)
    hdr = filter(l -> startswith(l, "#"), readlines(f))[end]
    names = [m.captures[1] for m in eachmatch(r"\[\d+\]=(\S+)", hdr)]
    d = readdlm(f, comments=true)
    Dict(n => d[:, i] for (i, n) in enumerate(names))
end
function alpha(run, t0, t1)
    h = hst(joinpath(RUN, run, "GravitoTurb.user.hst"))
    t = h["time"]; k = (t .>= t0) .& (t .<= t1)
    a = (2/3) .* (h["wgrv"] .+ h["wrey"]) ./ h["rho_cs2"]
    ar = (2/3) .* h["wrey"] ./ h["rho_cs2"]
    blk = floor.(Int, (t[k] .- t0) ./ 5)
    bm = [mean(a[k][blk .== i]) for i in unique(blk)]
    (a=mean(a[k]), err=std(bm)/sqrt(length(bm)), rey=mean(ar[k])/mean(a[k]))
end
function budget(f)
    s = read(f, String)
    blocks = split(s, r"\n(?=\S)")            # one block per run in a combined file
    out = Dict{Int,NamedTuple}()
    for b in blocks
        m = match(r"beta = ([\d.]+)", b); m === nothing && continue
        beta = round(Int, parse(Float64, m.captures[1]))
        g(key) = parse(Float64, match(Regex(key * raw".*?\(([+-][\d.]+) C\)"), b).captures[1])
        out[beta] = (S=g("stress work S"), Fz=g("z-face energy loss F_z"),
                     dE=g("dE_tot/dt"), R=g("residual R"))
    end
    out
end

# runs: (beta, run, window)
WZ = [(4, "gt_l64z16/b4lad_renorm_r4", 150, 300), (5, "gt_l64z16/b5lad_renorm_r4", 150, 300),
      (10, "gt_l64z16/b10ramp_renorm_r4", 75, 100), (20, "gt_l64z16/b20lad_renorm_r4", 150, 250),
      (40, "gt_l64z16/b40lad_renorm_r4", 350, 550), (80, "gt_l64z16/b80lad_renorm_r4", 750, 950)]
WIN = Dict(10 => (200, 300), 20 => (350, 450), 40 => (550, 750), 80 => (950, 1150))
PLM = [(b, "gt_sc14_stndrd_b$(b)_hse", WIN[b]...) for b in (10, 20, 40, 80)]
PPM = [(b, "gt_sc14_stndrd_b$(b)_hse_ppm4", WIN[b]...) for b in (10, 20, 40, 80)]
WZS = [(10, "gt_sc14_stndrd_b10_hse_wzfofc", 200, 300)]
meas(list) = [(b=b, alpha(r, t0, t1)...) for (b, r, t0, t1) in list]
wz, plm, ppm, wzs = meas(WZ), meas(PLM), meas(PPM), meas(WZS)
for (lab, v) in (("WENO-Z prod", wz), ("plm", plm), ("PPM4", ppm), ("WENO-Z std", wzs)), r in v
    @printf("%-12s beta=%2d alpha'=%.4f +- %.4f  ratio %.3f  Rey %.2f\n", lab, r.b, r.a, r.err,
            r.a/eq21(r.b), r.rey)
end

cW, cP, cQ = "#d7301f", :gray45, "#3182bd"
fig = Figure(size=(1400, 1050))
bb = exp10.(range(log10(3), log10(100), length=100))

a = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="β = Ω t_cool", ylabel="α′",
         title="(a) α′ against the cooling time", titlealign=:left,
         xticks=[4, 5, 10, 20, 40, 80])
lines!(a, bb, eq21.(bb), color=:black, linewidth=1.5, label="SC14 eq. 21: 4/[9γ(γ−1)β]")
function pts!(ax, v, col, mk, lab; ratio=false, open=false)
    x = [r.b for r in v]; y = [ratio ? r.a/eq21(r.b) : r.a for r in v]
    e = [ratio ? r.err/eq21(r.b) : r.err for r in v]
    errorbars!(ax, x, y, e, color=col, whiskerwidth=6)
    scatter!(ax, x, y, color=(open ? :white : col), strokecolor=col, strokewidth=1.5,
             marker=mk, markersize=13, label=lab)
end
pts!(a, plm, cP, :rect, "plm, standard box")
pts!(a, ppm, cQ, :utriangle, "PPM4, standard box")
pts!(a, wzs, cW, :circle, "WENO-Z+FOFC, standard box"; open=true)
pts!(a, wz, cW, :circle, "WENO-Z+FOFC, production box")
axislegend(a, position=:lb, framevisible=false, labelsize=12)

b = Axis(fig[1, 2], xscale=log10, xlabel="β", ylabel="α′ / eq. 21",
         title="(b) α′ relative to the thermal balance", titlealign=:left,
         xticks=[4, 5, 10, 20, 40, 80])
band!(b, [3.0, 100.0], [0.9, 0.9], [1.1, 1.1], color=(:gray, 0.15))
hlines!(b, [1.0], color=:black, linewidth=1)
pts!(b, plm, cP, :rect, "plm"; ratio=true)
pts!(b, ppm, cQ, :utriangle, "PPM4"; ratio=true)
pts!(b, wzs, cW, :circle, "WENO-Z, std"; ratio=true, open=true)
pts!(b, wz, cW, :circle, "WENO-Z, production"; ratio=true)
ylims!(b, 0.6, 1.6)
axislegend(b, position=:lt, framevisible=false, labelsize=12)

c = Axis(fig[2, 1], xscale=log10, xlabel="β", ylabel="Reynolds share of α′",
         title="(c) Reynolds stress / total stress", titlealign=:left, xticks=[4, 5, 10, 20, 40, 80])
hlines!(c, [0.0], color=:gray70, linestyle=:dot)
for (v, col, mk, lab, op) in ((plm, cP, :rect, "plm", false), (ppm, cQ, :utriangle, "PPM4", false),
                              (wzs, cW, :circle, "WENO-Z, std", true),
                              (wz, cW, :circle, "WENO-Z, production", false))
    scatterlines!(c, [r.b for r in v], [r.rey for r in v], color=col, marker=mk, markersize=12,
                  markercolor=(op ? :white : col), strokecolor=col, strokewidth=1.5, label=lab)
end
axislegend(c, position=:lb, framevisible=false, labelsize=12)

# energy budget
wb = Dict{Int,NamedTuple}()
for (bt, r, _, _) in WZ
    f = joinpath(RUN, r, "energy_budget.txt")
    isfile(f) && merge!(wb, budget(f))
end
pb = budget(joinpath(RUN, "gt_sc14_stndrd_b80_hse_ppm4", "energy_budget_b40_b80.txt"))
d = Axis(fig[2, 2], xscale=log10, xlabel="β", ylabel="fraction of the cooling C",
         title="(d) energy budget: dE/dt = S − C − F_z + R", titlealign=:left,
         xticks=[4, 5, 10, 20, 40, 80])
hlines!(d, [0.0, 1.0], color=:gray70, linestyle=:dot)
ks = sort(collect(keys(wb)))
scatterlines!(d, ks, [wb[k].S for k in ks], color=cW, marker=:circle, label="WENO-Z: stress work S")
scatterlines!(d, ks, [wb[k].S - wb[k].Fz for k in ks], color=cW, marker=:circle,
              markercolor=:white, strokecolor=cW, strokewidth=1.5, linestyle=:dash,
              label="WENO-Z: S − F_z")
scatterlines!(d, ks, [wb[k].Fz for k in ks], color="#fc8d59", marker=:diamond,
              label="WENO-Z: F_z (out through z faces)")
scatterlines!(d, ks, [wb[k].R for k in ks], color="#7a0177", marker=:xcross,
              label="WENO-Z: residual R")
kp = sort(collect(keys(pb)))
scatter!(d, kp, [pb[k].S for k in kp], color=cQ, marker=:utriangle, markersize=13,
         label="PPM4: S")
scatter!(d, kp, [pb[k].Fz for k in kp], color=cQ, marker=:dtriangle, markersize=13,
         label="PPM4: F_z")
plm80 = plm[end]
scatter!(d, [80.0], [1 - plm80.a/eq21(80)], color=cP, marker=:rect, markersize=13,
         label="plm: missing heating 1 − S/C")
ylims!(d, -0.1, 1.55)
axislegend(d, position=:lt, framevisible=false, labelsize=11, nbanks=2)

Label(fig[0, 1:2], "SC14 gravito-turbulence: α′ against β for plm, PPM4 and WENO-Z+FOFC (4 cells per H)",
      fontsize=18)
out = joinpath(@__DIR__, "alpha_beta.png")
save(out, fig, px_per_unit=1.5)
println("wrote ", out)
