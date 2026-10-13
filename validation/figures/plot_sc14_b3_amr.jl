# The standard-resolution beta = 3 AMR run: fragmentation from the beta = 10 state.
#
# Restarted from the beta = 10 restart at Omega t = 300 with bcool_beta = 3 and
# mesh_refinement/refinement=adaptive (via the gravito_turb_sc14_amr.athinput
# overlay).  Three hydro_w dumps exist: Omega t = 310, 320, 350.
#
#   top     (a) peak density, with the Jeans resolution of the peak annotated
#           (b) the two-phase sound speed: box average vs the same average with
#               the fragments (rho > 5) removed -- the box average is fragment-
#               dominated once they hold a third of the mass, which is why
#               <cs>_rho and Toomre Q stop describing the disk after ~Omega t 315
#           (c) mass budget: M/M0 and dM/dt, both from the history file
#   bottom  Sigma_g(x,y) at the three dump times, shared log color scale
#
# rho_max and the phase split are computed on the RAW MeshBlocks (peak density is
# lost to restriction); Sigma_g uses the conservatively restricted root grid, which
# leaves the column integral exact.  Continuous curves come from the history file
# at 0.25/Omega, markers from the three dumps.
#
# Usage: julia validation/figures/plot_sc14_b3_amr.jl
using CairoMakie, Printf, DelimitedFiles, Statistics
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))
include(joinpath(dirname(@__DIR__), "..", "scripts", "athenak_bin.jl"))

const HERE  = @__DIR__
const RUN   = joinpath(dirname(HERE), "run", "gt_sc14_stndrd_b3_amr")
const UNIF  = joinpath(dirname(HERE), "run", "gt_sc14_stndrd_b3_morphing")
const GAM   = 5/3
const G     = 0.3384
const SIG0  = 2.0
const RHO_F = 5.0        # the fragment / bulk-disk split used throughout

dumps = sort(filter(f -> occursin("hydro_w", f) && endswith(f, ".bin"),
                    readdir(RUN, join=true)))

# ---- per-dump reductions on the raw (unrestricted) blocks
times = Float64[]; rmax = Float64[]; lamJ = Float64[]
cs_bulk = Float64[]; cs_frag = Float64[]; fmass = Float64[]; nmb = Int[]
sigmas = Matrix{Float64}[]; xs = Any[]; ys = Any[]
for f in dumps
    fd  = read_bin(f)
    dx0 = (fd.x1max - fd.x1min)/fd.Nx1
    rm = 0.0; lj = 0.0
    mb = 0.0; mf = 0.0; mcb = 0.0; mcf = 0.0
    for m in 1:fd.n_mbs
        rho = fd.mb_data["dens"][m]
        ei  = fd.mb_data["eint"][m]
        lev = fd.mb_logical[m, 4]
        dx  = dx0/2^lev; dV = dx^3
        cs  = sqrt.(GAM*(GAM-1).*ei./rho)
        i   = argmax(rho)
        if rho[i] > rm
            rm = rho[i]
            lj = cs[i]*sqrt(pi/(G*rho[i]))/dx
        end
        sel = rho .< RHO_F
        mb  += sum(rho[sel])*dV;      mcb += sum(rho[sel].*cs[sel])*dV
        mf  += sum(rho[.!sel])*dV;    mcf += sum(rho[.!sel].*cs[.!sel])*dV
    end
    push!(times, fd.time); push!(rmax, rm); push!(lamJ, lj); push!(nmb, fd.n_mbs)
    push!(cs_bulk, mcb/mb); push!(cs_frag, mf > 0 ? mcf/mf : NaN)
    push!(fmass, mf/(mb+mf))
    r3 = assemble_root(fd, "dens")
    dz = (fd.x3max - fd.x3min)/fd.Nx3
    push!(sigmas, dropdims(sum(r3, dims=3), dims=3) .* dz ./ SIG0)
    push!(xs, range(fd.x1min, fd.x1max, length=fd.Nx1))
    push!(ys, range(fd.x2min, fd.x2max, length=fd.Nx2))
    @printf("t=%3.0f  nmb=%4d  rho_max=%8.1f  lamJ/dx=%5.2f  cs(bulk)=%5.2f  cs(frag)=%6.2f  f_mass=%.3f\n",
            fd.time, fd.n_mbs, rm, lj, mcb/mb, mcf/mf, mf/(mb+mf))
end

# ---- history file
function hist(dir)
    d = readdlm(only(filter(x -> endswith(x, ".user.hst"), readdir(dir, join=true))),
                comments=true, comment_char='#')
    keep = falses(size(d,1)); tmin = Inf
    for i in size(d,1):-1:1
        d[i,1] < tmin && (keep[i] = true; tmin = d[i,1])
    end
    d = d[keep, :]
    (t=d[:,1], M=d[:,3], cs=d[:,4]./d[:,3])
end
h = hist(RUN); hu = hist(UNIF)
dMdt = [ (m = abs.(h.t .- h.t[i]) .<= 2.0;
          sum(m) < 3 ? NaN :
          ((tt = h.t[m]; yy = h.M[m]);
           sum((tt .- mean(tt)).*(yy .- mean(yy)))/sum((tt .- mean(tt)).^2)))
         for i in eachindex(h.t) ]

const CFRAG = RGBf(0.80, 0.14, 0.22)
const CBULK = RGBf(0.17, 0.42, 0.72)
const CGREY = RGBf(0.45, 0.45, 0.45)

fig = Figure(size=(1500, 1010))
Label(fig[0, 1:3],
      "β = 3 from the β = 10 state at Ωt = 300, standard resolution + AMR: fragmentation into two bound clumps",
      fontsize=19, font=:bold, padding=(0, 0, 6, 0))

# ---------------------------------------------------------------- (a) rho_max
a1 = Axis(fig[1, 1]; xlabel="Ω t", ylabel="ρ_max / ρ₀", yscale=log10,
          title="(a)  peak density", titlealign=:left, titlefont=:bold, titlesize=15,
          yticks=([10,30,100,300,1000], ["10","30","100","300","1000"]),
          xgridcolor=(:gray,0.15), ygridcolor=(:gray,0.15))
lines!(a1, times, rmax, color=CFRAG, linewidth=2.2)
scatter!(a1, times, rmax, color=CFRAG, markersize=13, strokecolor=:white, strokewidth=1)
hlines!(a1, [11.2], color=(CGREY,0.9), linestyle=:dash)
text!(a1, 349, 12.5; text="uniform β = 3, no AMR (Ωt = 330)", align=(:right,:bottom),
      color=CGREY, fontsize=11)
for i in eachindex(times)   # first point sits on the rising line: label it below
    if i == 1
        text!(a1, times[i]+1.0, rmax[i]/1.3; text=@sprintf("λ_J/Δx = %.1f", lamJ[i]),
              align=(:left,:top), color=CFRAG, fontsize=10.5)
    else
        text!(a1, times[i], rmax[i]*1.4; text=@sprintf("λ_J/Δx\n= %.1f", lamJ[i]),
              align=(:center,:bottom), color=CFRAG, fontsize=10.5)
    end
end
xlims!(a1, 302, 353); ylims!(a1, 7, 6000)

# ---------------------------------------------------------------- (b) two-phase cs
a2 = Axis(fig[1, 2]; xlabel="Ω t", ylabel="⟨c_s⟩_ρ", yscale=log10,
          title="(b)  the box average is fragment-dominated",
          titlealign=:left, titlefont=:bold, titlesize=15,
          yticks=([2,3,5,10,20], ["2","3","5","10","20"]),
          xgridcolor=(:gray,0.15), ygridcolor=(:gray,0.15))
lines!(a2, h.t, h.cs, color=:black, linewidth=2.0, label="whole box (history)")
scatter!(a2, times, cs_frag, color=CFRAG, markersize=13, strokecolor=:white, strokewidth=1,
         label="fragments, ρ > 5ρ₀")
lines!(a2, times, cs_frag, color=CFRAG, linewidth=1.8)
scatter!(a2, times, cs_bulk, color=CBULK, markersize=13, strokecolor=:white, strokewidth=1,
         label="bulk disk, ρ < 5ρ₀")
lines!(a2, times, cs_bulk, color=CBULK, linewidth=1.8)
lines!(a2, hu.t, hu.cs, color=(CGREY,0.85), linewidth=1.6, linestyle=:dash,
       label="uniform β = 3, whole box")
xlims!(a2, 302, 353); ylims!(a2, 1.7, 22)
axislegend(a2, position=:lt, framevisible=false, labelsize=11)

# ---------------------------------------------------------------- (c) mass budget
a3 = Axis(fig[1, 3]; xlabel="Ω t", ylabel="M / M₀",
          title="(c)  a quarter of the disk leaves through the z faces",
          titlealign=:left, titlefont=:bold, titlesize=15,
          xgridcolor=(:gray,0.15), ygridcolor=(:gray,0.15))
a3r = Axis(fig[1, 3]; ylabel="dM/dt", yaxisposition=:right, ygridvisible=false,
           xgridvisible=false, yticklabelcolor=CBULK, ylabelcolor=CBULK)
hidespines!(a3r); hidexdecorations!(a3r)
linkxaxes!(a3, a3r)
lines!(a3r, h.t, dMdt, color=(CBULK,0.85), linewidth=1.8)
hlines!(a3r, [0.0], color=(CBULK,0.35), linewidth=1)
lines!(a3, h.t, h.M ./ h.M[1], color=:black, linewidth=2.4)
lines!(a3, hu.t, hu.M ./ hu.M[1], color=(CGREY,0.85), linewidth=1.6, linestyle=:dash)
scatter!(a3, times, [h.M[argmin(abs.(h.t .- tt))]/h.M[1] for tt in times],
         color=:black, markersize=10)
text!(a3, 331, 0.947; text="uniform β = 3", align=(:right,:top), color=CGREY, fontsize=11)
text!(a3, 306, 0.76; text="M/M₀", align=(:left,:bottom), color=:black, fontsize=12)
text!(a3, 345, 0.905; text="dM/dt", align=(:left,:bottom), color=CBULK, fontsize=12)
xlims!(a3, 302, 353); xlims!(a3r, 302, 353)
ylims!(a3, 0.70, 1.02); ylims!(a3r, -120, 20)

# ---------------------------------------------------------------- Sigma maps
vmin, vmax = -1.6, 2.7
local hm
for (n, tt) in enumerate(times)
    ax = Axis(fig[2, n]; aspect=DataAspect(),
              title=@sprintf("Ωt = %.0f      %d MeshBlocks", tt, nmb[n]),
              titlealign=:left, titlefont=:bold, titlesize=15,
              xlabel="x / H", ylabel=(n == 1 ? "y / H" : ""),
              yticklabelsvisible=(n == 1),
              xticks=-32:16:32, yticks=-32:16:32)
    global hm = heatmap!(ax, xs[n], ys[n], log10.(clamp.(sigmas[n], 1e-10, Inf));
                         colormap=:inferno, colorrange=(vmin, vmax))
    lab = @sprintf("Σ_max = %.0f Σ₀\nρ>5ρ₀ holds %.0f%% of the mass",
                   maximum(sigmas[n]), 100*fmass[n])
    poly!(ax, Rect2f(-31.0, -31.0, 30.0, 6.4); color=(:black, 0.55),
          strokecolor=(:white, 0.25), strokewidth=0.8)
    text!(ax, -30.0, -30.0; text=lab, align=(:left, :bottom), color=:white, fontsize=12)
end
Colorbar(fig[2, 4], hm; label="log₁₀ Σ_g / Σ₀", height=Relative(0.9), width=16,
         labelsize=17)

Label(fig[3, 1:3],
      "Fragments are adiabatic, not under-resolved: t_ff = 0.03/Ω at ρ = 10³ is ~100× shorter than t_cool = 3/Ω, so a collapsing clump cannot radiate and heats until\n" *
      "pressure halts it — the measured c_s sits between isothermal and the fully adiabatic 2.2·ρ^(1/3), and λ_J stays above 6 cells throughout (Truelove needs 4).\n" *
      "AMR reaches only one level (8 cells/H): with 3 root MeshBlocks in z the slab-open x3 policy forbids any level ≥ 2, whatever num_levels says.",
      fontsize=12, color=:gray30, justification=:left, lineheight=1.3,
      padding=(0, 0, 4, 4))
rowgap!(fig.layout, 8); colgap!(fig.layout, 12)

out = joinpath(HERE, "sc14_b3_amr.png")
save(out, fig; px_per_unit=2)
println("wrote $out")
