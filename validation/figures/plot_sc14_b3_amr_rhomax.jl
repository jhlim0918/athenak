# rho_max(t) through the beta = 3 fragmentation, against the free-fall time.
#
# From the 101 hydro_w dumps (0.25/Omega) of the AMR rerun (Omega t = 300-325):
#   (a) peak density on the raw (unrestricted) MeshBlocks
#   (b) timescales: the free-fall time at the peak, t_ff = sqrt(3 pi / 32 G rho_max),
#       the measured growth time tau = (d ln rho_max / dt)^-1 (centered difference,
#       1/Omega boxcar), and the imposed cooling time t_cool = beta/Omega = 3.
#       A collapse running on free fall has tau ~ t_ff; tau >> t_ff means the clump
#       is contracting quasi-statically, held up by pressure.  t_ff << t_cool is the
#       adiabatic regime where the clump heats as it collapses.
#   (c) Jeans resolution at the peak, lambda_J / dx (Truelove et al. 1997 need >= 4).
#
# Usage: julia validation/figures/plot_sc14_b3_amr_rhomax.jl
using CairoMakie, Printf, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
include(joinpath(dirname(@__DIR__), "..", "scripts", "athenak_bin.jl"))

const HERE = @__DIR__
const BDIR = joinpath(dirname(HERE), "run", "gt_sc14_stndrd_b3_amr", "bin")
const G = 0.3384; const GAM = 5/3; const BETA = 3.0
tff(rho) = sqrt(3pi/(32*G*rho))

files = sort(filter(f -> occursin("hydro_w", f) && endswith(f, ".bin"), readdir(BDIR, join=true)))
t = Float64[]; rmax = Float64[]; csp = Float64[]; lamJ = Float64[]; levp = Int[]
for (n, f) in enumerate(files)
    fd = read_bin(f); dx0 = (fd.x1max - fd.x1min)/fd.Nx1
    rm = 0.0; cs_at = 0.0; lj = 0.0; lv = 0
    for m in 1:fd.n_mbs
        rho = fd.mb_data["dens"][m]; i = argmax(rho)
        rho[i] > rm || continue
        lev = fd.mb_logical[m, 4]; dx = dx0/2^lev
        cs = sqrt(GAM*(GAM-1)*fd.mb_data["eint"][m][i]/rho[i])
        rm = rho[i]; cs_at = cs; lj = cs*sqrt(pi/(G*rho[i]))/dx; lv = lev
    end
    push!(t, fd.time); push!(rmax, rm); push!(csp, cs_at); push!(lamJ, lj); push!(levp, lv)
    n % 20 == 0 && @printf("  %3d/%d  Ωt=%6.2f  rho_max=%8.2f  cs=%5.2f  λ_J/Δx=%5.2f  level=%d\n",
                           n, length(files), fd.time, rm, cs_at, lj, lv)
end

# growth time from a centered difference of ln rho_max, boxcar 1/Omega (5 samples)
lnr = log.(rmax)
dlnr = similar(lnr)
for i in eachindex(lnr)
    lo = max(1, i-2); hi = min(length(lnr), i+2)
    dlnr[i] = (lnr[hi] - lnr[lo])/(t[hi] - t[lo])
end
tau = 1.0 ./ dlnr                       # >0 while rising
tf  = tff.(rmax)

@printf("\nrho_max: %.2f (t=%.0f) -> %.1f (t=%.2f) -> %.1f (t=%.0f)\n",
        rmax[1], t[1], maximum(rmax), t[argmax(rmax)], rmax[end], t[end])
@printf("min lambda_J/dx at the peak = %.2f at t=%.2f (rho=%.0f)\n",
        minimum(lamJ), t[argmin(lamJ)], rmax[argmin(lamJ)])
rising = tau .> 0
@printf("median tau/t_ff while rising = %.1f ; min = %.1f\n",
        median((tau./tf)[rising]), minimum((tau./tf)[rising]))
i1 = findfirst(rmax .> 10); i2 = findfirst(rmax .> 100); i3 = findfirst(rmax .> 500)
for (lab, i) in (("rho>10", i1), ("rho>100", i2), ("rho>500", i3))
    i === nothing || @printf("  first %-8s at Ωt = %.2f\n", lab, t[i])
end

const CR = RGBf(0.80,0.14,0.22); const CB = RGBf(0.17,0.42,0.72); const CG = RGBf(0.13,0.55,0.35)
fig = Figure(size=(900, 1040))
Label(fig[0,1], "β = 3 with AMR, restarted from β = 10 at Ωt = 300: the collapse at 0.25/Ω cadence",
      fontsize=17, font=:bold, padding=(0,0,4,0))

a1 = Axis(fig[1,1]; ylabel="ρ_max / ρ₀", yscale=log10, xticklabelsvisible=false,
          yticks=([1,3,10,30,100,300,1000],["1","3","10","30","100","300","1000"]),
          title="(a)  peak density", titlealign=:left, titlefont=:bold, titlesize=15,
          xgridcolor=(:gray,0.15), ygridcolor=(:gray,0.15))
lines!(a1, t, rmax, color=CR, linewidth=2.2)
scatter!(a1, t, rmax, color=CR, markersize=5)
hlines!(a1, [5.0], color=(:gray,0.7), linestyle=:dot)
text!(a1, 324.8, 5.3; text="AMR criterion, ρ > 5", align=(:right,:bottom), color=:gray, fontsize=11)
for (tt, rr) in ((310.0, 90.8), (320.0, 990.8))
    scatter!(a1, [tt], [rr], color=:white, strokecolor=:black, strokewidth=1.6, markersize=11)
end
text!(a1, 311.0, 450.0; text="○ first run, 10/Ω dumps", align=(:left,:bottom), color=:gray30, fontsize=10.5)
ylims!(a1, 1.5, 3000)

a2 = Axis(fig[2,1]; ylabel="timescale  (Ω⁻¹)", yscale=log10, xticklabelsvisible=false,
          yticks=([0.03,0.1,0.3,1,3,10,30],["0.03","0.1","0.3","1","3","10","30"]),
          title="(b)  free-fall vs measured growth vs cooling", titlealign=:left,
          titlefont=:bold, titlesize=15, xgridcolor=(:gray,0.15), ygridcolor=(:gray,0.15))
lines!(a2, t, tf, color=CB, linewidth=2.2, label="t_ff(ρ_max) = √(3π / 32Gρ_max)")
lines!(a2, t[rising], tau[rising], color=CG, linewidth=2.0,
       label="τ_growth = (d ln ρ_max/dt)⁻¹, 1/Ω boxcar (rising phases)")
scatter!(a2, t[rising], tau[rising], color=CG, markersize=4)
hlines!(a2, [BETA], color=:black, linestyle=:dash, linewidth=1.8, label="t_cool = β/Ω = 3")
hlines!(a2, [1.0], color=(:gray,0.7), linestyle=:dot)
text!(a2, 324.8, 1.0; text="1/Ω", align=(:right,:bottom), color=:gray, fontsize=11)
ylims!(a2, 0.02, 60)
axislegend(a2, position=:lt, framevisible=false, labelsize=11, rowgap=-2)

a3 = Axis(fig[3,1]; xlabel="Ω t", ylabel="λ_J / Δx at the peak", yscale=log10,
          yticks=([3,4,6,10,20,40],["3","4","6","10","20","40"]),
          title="(c)  Jeans resolution of the peak cell", titlealign=:left,
          titlefont=:bold, titlesize=15, xgridcolor=(:gray,0.15), ygridcolor=(:gray,0.15))
lines!(a3, t, lamJ, color=:black, linewidth=2.0)
scatter!(a3, t, lamJ, color=[l == 1 ? CR : (:gray,0.6) for l in levp], markersize=5)
hlines!(a3, [4.0], color=CR, linestyle=:dash, linewidth=1.6)
text!(a3, 300.5, 4.2; text="Truelove: 4 cells", align=(:left,:bottom), color=CR, fontsize=11)
text!(a3, 324.8, 35; text="dots red when the peak sits on a level-1 block (8 cells/H)",
      align=(:right,:top), color=:gray30, fontsize=10.5)
ylims!(a3, 2.5, 50)

linkxaxes!(a1, a2, a3)
xlims!(a1, 299.8, 325.2)
Label(fig[4,1],
      "t_ff drops below t_cool almost immediately, so the clumps are adiabatic throughout; the measured e-folding time stays well above t_ff — a pressure-regulated,\n" *
      "quasi-static contraction rather than free fall.  Growth time is undefined (omitted) where ρ_max is falling or flat.  Free-fall time uses G = 0.3384 (SC14 units).",
      fontsize=11.5, color=:gray30, justification=:left, lineheight=1.3, padding=(0,0,4,4))
rowgap!(fig.layout, 6)
out = joinpath(HERE, "sc14_b3_amr_rhomax.png")
save(out, fig; px_per_unit=2); println("wrote $out")
