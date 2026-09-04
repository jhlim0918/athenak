# Phase 4d: the "drag-coupling instability" of Phase 4c resolved.  Left: vertical kinetic
# energy of the 3D shear-periodic NSH box with NO particles, gas velocities seeded with
# 1e-10 white noise, at several Courant numbers under imex2+ and rk2: the growth at
# cfl >= 0.4 and the decay at 0.3 (and at nz = 8) are the hydro scheme's own 3D grid
# checkerboard mode.  Right: the power spectrum of the gas v_z of the unstable dust run
# (32^3, cfl 0.45) at t = 2.0 along the diagonal (k,k,k): the growing mode sits at the
# grid Nyquist.
#
# Data: validation/run/dust4d/inst/ (README in validation/run/dust4d/)
# Usage: julia validation/figures/plot_dust4d_checkerboard.jl
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=15))
const HERE = @__DIR__
const RUN  = joinpath(dirname(HERE), "run", "dust4d", "inst")
hst(path) = readdlm(path, comments=true, comment_char='#')

fig = Figure(size=(1200, 440))
ax1 = Axis(fig[1, 1], xlabel="t  (Ω⁻¹)", ylabel="gas vertical kinetic energy", yscale=log10,
           title="gas only, seeded at 1e-10 (32³ shear-periodic NSH box)")
for (r, lab, ls) in (("imex_c45", "imex2+, cfl 0.45", :solid), ("rk2_c45", "rk2, cfl 0.45", :dash),
                     ("rk2_c40", "rk2, cfl 0.40", :dashdot), ("imex_c30", "imex2+, cfl 0.30", :solid),
                     ("rk2_c30", "rk2, cfl 0.30", :dash), ("imex_c45_nz8", "imex2+, cfl 0.45, nz = 8", :dot))
    d = hst(joinpath(RUN, "gasonly_seeded_" * r * ".hydro.hst"))
    lines!(ax1, d[:, 1], max.(d[:, 9], 1e-30), label=lab, linestyle=ls, linewidth=2)
end
axislegend(ax1, position=:rc, framevisible=false)

# spectrum from the archived eigenmode listing: parse the top modes line at t ~ 2.0
lines_ = readlines(joinpath(RUN, "eigenmode_spectrum.txt"))
ax2 = Axis(fig[1, 2], xlabel="t  (Ω⁻¹)", ylabel="rms of gas v_z (unstable dust run)", yscale=log10,
           title="dust run, cfl 0.45: the mode at (kx,ky,kz) ≈ (16,15,15) of 32")
ts = Float64[]; rms = Float64[]
for l in lines_
    m = match(r"^t=([\d.]+)\s+vz: rms=([\d.e+-]+)", l)
    m === nothing && continue
    push!(ts, parse(Float64, m.captures[1])); push!(rms, parse(Float64, m.captures[2]))
end
scatterlines!(ax2, ts, max.(rms, 1e-30), markersize=12, label="rms v_z (bin outputs)")
i0 = findfirst(t -> t > 1.1, ts)
lines!(ax2, ts[i0:i0+4], rms[i0] .* exp.(35 .* (ts[i0:i0+4] .- ts[i0])), color=:gray, linestyle=:dash, linewidth=2, label="e^{35 t}")
axislegend(ax2, position=:rb, framevisible=false)
save(joinpath(HERE, "dust4d_checkerboard.png"), fig, px_per_unit=2)
println("wrote dust4d_checkerboard.png")
