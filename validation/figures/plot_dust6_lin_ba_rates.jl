# Growth rate of the BA-parameter linear mode (tau_s = 1, eps = 0.2, Kx = Kz = 0.86,
# s = 0.094720 analytic) against the cells per wavelength, on uniform meshes and on refined
# meshes (root at half, the middle half of the box refined once): left the rate itself,
# right the error 1 - s/s_analytic on log axes with a second-order reference.
#   DUST6_RUN=<dir with the dust6 runs> julia plot_dust6_lin_ba_rates.jl   (reads lin_BA/<run>/)
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png"); set_theme!(Theme(fontsize=15))
const HERE = @__DIR__
const RUN  = joinpath(get(ENV, "DUST6_RUN", joinpath(dirname(HERE), "run", "dust6")), "lin_BA")
const SREF = 0.094720; const T0, T1 = 20.0, 90.0
labs(p) = (for l in eachline(p); occursin("[1]=", l) && return [split(t, "=")[2] for t in split(replace(l, "#" => "")) if occursin("=", t)]; end; String[])
function rate(r)
    p = joinpath(RUN, r, "lin_$r.user.hst"); isfile(p) || return NaN
    h = readdlm(p, comments=true, comment_char='#'); L = labs(p); t = h[:,1]; a = h[:, findfirst(==("dp_max"), L)]
    m = (t .>= T0) .& (t .<= T1) .& (a .> 0); X = hcat(ones(sum(m)), t[m]); (X \ log.(a[m]))[2]
end
Nu = [8, 16, 32, 64]; su = [rate("u$N") for N in Nu]
Nr = [16, 32, 64];    sr = [rate("r$N") for N in Nr]
fig = Figure(size=(1150, 470))
ax1 = Axis(fig[1,1], xlabel="cells per wavelength (finest level)", ylabel="growth rate s  (Ω)", xscale=log2,
           title="BA mode, Kx = Kz = 0.86")
hlines!(ax1, [SREF], color=:black, linestyle=:dash, linewidth=1.5, label=@sprintf("analytic s = %.5f", SREF))
scatterlines!(ax1, Nu, su, color=:black, marker=:circle, markersize=15, linewidth=2, label="uniform mesh")
scatterlines!(ax1, Nr, sr, color=:crimson, marker=:utriangle, markersize=16, linewidth=2, linestyle=:dash,
              label="refined: root at N/2, half the box at N")
for (N, s) in zip(Nu, su); text!(ax1, N, s, text=@sprintf("%.4f", s), align=(:center, :top), offset=(0, -8), fontsize=11); end
for (N, s) in zip(Nr, sr); text!(ax1, N, s, text=@sprintf("%.4f", s), align=(:center, :bottom), offset=(0, 8), fontsize=11, color=:crimson); end
axislegend(ax1, position=:rb, framevisible=false, labelsize=12)
ax1.xticks = (Nu, string.(Nu)); ylims!(ax1, 0.06, 0.1)
ax2 = Axis(fig[1,2], xlabel="cells per wavelength (finest level)", ylabel="1 − s / s_analytic", xscale=log2, yscale=log10,
           title="error, with an N⁻² reference")
scatterlines!(ax2, Nu, 1 .- su ./ SREF, color=:black, marker=:circle, markersize=15, linewidth=2, label="uniform")
scatterlines!(ax2, Nr, 1 .- sr ./ SREF, color=:crimson, marker=:utriangle, markersize=16, linewidth=2, linestyle=:dash, label="refined")
Nn = [12.0, 80.0]; e16 = 1 - su[2]/SREF
lines!(ax2, Nn, e16 .* (16 ./ Nn).^2, color=:gray50, linestyle=:dot, linewidth=1.5, label="∝ N⁻²")
axislegend(ax2, position=:lb, framevisible=false, labelsize=12)
ax2.xticks = (Nu, string.(Nu))
out = joinpath(HERE, "dust6_lin_ba_rates.png"); save(out, fig, px_per_unit=1.6); println("wrote ", out)
