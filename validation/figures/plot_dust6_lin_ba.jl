# Linear growth of the BA-parameter streaming mode (tau_s = 1, eps = 0.2, Kx = Kz = 0.86,
# s = 0.094720 from the YG05 dispersion relation, scripts/analysis/si_linear.py) on uniform
# meshes of 8, 16, 32, 64 cells per wavelength and on refined meshes whose root is half the
# finest (the middle half of a 4-wavelength box refined once): the amplitude histories and
# the measured growth rate against the cells per wavelength.
#   DUST6_RUN=<dir with the dust6 runs> julia plot_dust6_lin_ba.jl     (reads lin_BA/<run>/)
using CairoMakie, DelimitedFiles, Printf
CairoMakie.activate!(type="png"); set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
const RUN  = joinpath(get(ENV, "DUST6_RUN", joinpath(dirname(HERE), "run", "dust6")), "lin_BA")
const SREF = 0.094720; const T0, T1 = 20.0, 90.0
labs(p) = (for l in eachline(p); occursin("[1]=", l) && return [split(t, "=")[2] for t in split(replace(l, "#" => "")) if occursin("=", t)]; end; String[])
function rate(p)
    h = readdlm(p, comments=true, comment_char='#'); L = labs(p); t = h[:,1]; a = h[:, findfirst(==("dp_max"), L)]
    m = (t .>= T0) .& (t .<= T1) .& (a .> 0); X = hcat(ones(sum(m)), t[m]); (t, a, (X \ log.(a[m]))[2])
end
uni = [(8, "u8"), (16, "u16"), (32, "u32"), (64, "u64")]
ref = [(16, "r16"), (32, "r32"), (64, "r64")]
fig = Figure(size=(1250, 480))
ax1 = Axis(fig[1,1], xlabel="t  (Ω⁻¹)", ylabel="max |δρ_p| / ρ_p", yscale=log10,
           title="BA mode, K = 0.86: dust-density amplitude")
ax2 = Axis(fig[1,2], xlabel="cells per wavelength (finest level)", ylabel="s / s_analytic", xscale=log2,
           title="growth rate against resolution")
cols = Dict(8 => :gray40, 16 => :steelblue, 32 => :darkorange, 64 => :crimson)
su = Float64[]; sr = Float64[]
for (N, r) in uni
    p = joinpath(RUN, r, "lin_$r.user.hst"); isfile(p) || (push!(su, NaN); continue)
    t, a, s = rate(p); push!(su, s)
    lines!(ax1, t, a ./ 0.2, color=cols[N], linewidth=2, label=@sprintf("uniform %d: s = %.4f", N, s))
end
for (N, r) in ref
    p = joinpath(RUN, r, "lin_$r.user.hst"); isfile(p) || (push!(sr, NaN); continue)
    t, a, s = rate(p); push!(sr, s)
    lines!(ax1, t, a ./ 0.2, color=cols[N], linewidth=2, linestyle=:dash, label=@sprintf("refined %d→%d: s = %.4f", N ÷ 2, N, s))
end
tt = range(0, 100, length=50); lines!(ax1, tt, 1e-6 .* exp.(SREF .* tt), color=:black, linestyle=:dot, linewidth=1.5, label="analytic s = 0.0947")
axislegend(ax1, position=:lt, framevisible=false, labelsize=11)
hlines!(ax2, [1.0], color=:black, linestyle=:dot)
scatterlines!(ax2, first.(uni), su ./ SREF, color=:black, marker=:circle, markersize=14, linewidth=2, label="uniform")
scatterlines!(ax2, first.(ref), sr ./ SREF, color=:crimson, marker=:utriangle, markersize=14, linewidth=2, linestyle=:dash,
              label="refined: root N/2, half the box at N")
axislegend(ax2, position=:rb, framevisible=false, labelsize=11)
ax2.xticks = ([8, 16, 32, 64], ["8", "16", "32", "64"])
out = joinpath(HERE, "dust6_lin_ba.png"); save(out, fig, px_per_unit=1.5); println("wrote ", out)
