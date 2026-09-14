# Time series of the JY07 run BA on the uniform mesh and on the three adaptive meshes
# (validation/run/dust6/BA_u256, BA_amr3c, BA_amr4, BA_amr4_mb8): peak particle-mesh dust
# density, radial velocity dispersion of the dust, and the particle count relative to the
# initial 262,144 -- the last panel is where the 8^2-block run shows its uninitialised
# sampling level (validation doc, sec. 12).  DUST6_RUN=<dir with the dust6 runs>.
using CairoMakie, DelimitedFiles, Printf, Statistics
CairoMakie.activate!(type="png"); set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
const RUN  = get(ENV, "DUST6_RUN", joinpath(dirname(HERE), "run", "dust6"))
hst(p) = readdlm(p, comments=true, comment_char='#')
labs(p) = (for l in eachline(p); occursin("[1]=", l) && return [split(t, "=")[2] for t in split(replace(l, "#" => "")) if occursin("=", t)]; end; String[])
runs = [("BA_u256/si_BA.phst",        "uniform 256²",              :black,      2.6, :solid),
        ("BA_amr3c/si_BA3.phst",      "AMR 3 levels, 16² blocks",  :steelblue,  1.8, :dash),
        ("BA_amr4/si_BA4.phst",       "AMR 4 levels, 16² blocks",  :darkorange, 1.8, :dashdot),
        ("BA_amr4_mb8/si_BA4m8.phst", "AMR 4 levels,  8² blocks, defective PLEV", :gray50, 1.4, :dash),
        ("BA_amr4_mb8_fix/si_BA4m8.phst", "AMR 4 levels,  8² blocks",  :crimson,    2.0, :solid)]
fig = Figure(size=(1500, 460))
ax1 = Axis(fig[1,1], xlabel="t  (Ω⁻¹)", ylabel="max ρ_p / ⟨ρ_p⟩", yscale=log10, title="peak dust density")
ax2 = Axis(fig[1,2], xlabel="t  (Ω⁻¹)", ylabel="σ(v_px)  (c_s)", title="radial velocity dispersion of the dust")
ax3 = Axis(fig[1,3], xlabel="t  (Ω⁻¹)", ylabel="N / N(t=0)", title="particle count (N(0) = 262,144)")
for (f, l, c, w, ls) in runs
    p = joinpath(RUN, f); isfile(p) || (println("missing ", f); continue)
    h = hst(p); L = labs(p); col = Dict(x => i for (i, x) in enumerate(L)); t = h[:,col["time"]]
    lines!(ax1, t, h[:,col["dpm_max"]] ./ 0.2, color=c, linewidth=w, linestyle=ls, label=l)
    lines!(ax2, t, h[:,col["sig_vx_1"]], color=c, linewidth=w, linestyle=ls, label=l)
    lines!(ax3, t, h[:,col["npart"]] ./ h[1,col["npart"]], color=c, linewidth=w, linestyle=ls, label=l)
end
vlines!(ax1, [100.0], color=(:gray, 0.5), linestyle=:dot); text!(ax1, 104, 1.6, text="hold ends", color=:gray, fontsize=11)
axislegend(ax3, position=:lt, framevisible=false, labelsize=11)
for ax in (ax1, ax2, ax3); xlims!(ax, 0, 500); end
out = joinpath(HERE, "dust6_ba_hist.png")
save(out, fig, px_per_unit=1.5)
println("wrote ", out)
