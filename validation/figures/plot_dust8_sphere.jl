# dust8: pressureless collapse of a uniform sphere of dust particles (Simon et al. 2016,
# Sect. 2.2.1 and Fig. 1), uniform 64^3/128^3/256^3 and AMR from a 64^3 root to 256^3:
#   (a) mass-weighted rms radius and the outermost particle's radius over their initial values
#   (b) maximum particle-mesh density / rho_p0
#   (c) relative error of the rms radius against r/r0 = cos^2 a
#   (d) relative error of the maximum density against cos^-6 a
# Analytic: a + sin(2a)/2 = (pi/2) t/t_ff, t_ff = sqrt(3 pi/(32 G rho_p0)).
#   julia plot_dust8_sphere.jl <run_root> out.png name:label ...
using CairoMakie, Printf
root, out = ARGS[1:2]; runs = [split(a, ":"; limit=2) for a in ARGS[3:end]]
G = 8e-3; rho0 = 1.0; tff = sqrt(3pi/(32G*rho0))
function alpha(tau)               # a + sin(2a)/2 = tau, 0 <= tau < pi/2
    a = tau
    for _ in 1:60
        f = a + 0.5sin(2a) - tau; fp = 1 + cos(2a)
        a = clamp(a - f/max(fp, 1e-12), 0, pi/2 - 1e-12)
    end
    a
end
cos2(t) = cos(alpha(pi/2*t/tff))^2
rd(f) = reduce(hcat, [parse.(Float64, split(l)) for l in eachline(f) if !startswith(l, "#") && !isempty(strip(l))])'
fig = Figure(size=(1400, 950), fontsize=16)
axa = Axis(fig[1, 1], xlabel=L"t/t_{\mathrm{ff}}", ylabel=L"r/r_0", title="(a) radius: rms (solid), outermost particle (dashed)")
axb = Axis(fig[1, 2], yscale=log10, xlabel=L"t/t_{\mathrm{ff}}", ylabel=L"\max\,\rho_p/\rho_{p0}", title="(b) maximum particle density")
axc = Axis(fig[2, 1], yscale=log10, xlabel=L"t/t_{\mathrm{ff}}", ylabel=L"|r_{\mathrm{rms}}/r_{\mathrm{ana}} - 1|", title="(c) error of the rms radius")
axd = Axis(fig[2, 2], yscale=log10, xlabel=L"t/t_{\mathrm{ff}}", ylabel=L"|\rho_{\max}/\rho_{\mathrm{ana}} - 1|", title="(d) error of the maximum density")
ts = range(0, 0.97, length=400)
lines!(axa, ts, cos2.(ts .* tff), color=:black, linewidth=3, label="analytic")
lines!(axb, ts, cos2.(ts .* tff) .^ -3, color=:black, linewidth=3, label="analytic")
cols = Makie.wong_colors()
for (n, (name, lab)) in enumerate(runs)
    d = joinpath(root, name); isfile(joinpath(d, "sphere.user.hst")) || continue
    u = rd(joinpath(d, "sphere.user.hst")); p = rd(joinpath(d, "sphere.phst"))
    t = u[:, 1] ./ tff; rr = sqrt.(u[:, 4] ./ u[:, 3]); rm = u[:, 5]
    ana = cos2.(u[:, 1])
    c = cols[mod1(n, length(cols))]
    lines!(axa, t, rr ./ rr[1], color=c, label=lab); lines!(axa, t, rm ./ rm[1], color=c, linestyle=:dash)
    tp = p[:, 1] ./ tff; rp = p[:, 12]
    lines!(axb, tp, rp ./ rho0, color=c, label=lab)
    sel = t .> 0.02
    lines!(axc, t[sel], abs.(rr[sel] ./ rr[1] ./ ana[sel] .- 1) .+ 1e-6, color=c, label=lab)
    selp = tp .> 0.02
    lines!(axd, tp[selp], abs.(rp[selp] ./ cos2.(p[selp, 1]) .^ -3 .- 1) .+ 1e-6, color=c, label=lab)
    @printf("%-8s t/tff %.3f: r_rms/r0 %.4f (ana %.4f), r_max/r0 %.4f, rho_max %.3g (ana %.3g); at 0.90 t_ff: r_rms err %.2e, rho err %.2e\n",
            name, t[end], rr[end]/rr[1], ana[end], rm[end]/rm[1], rp[end], cos2(p[end, 1])^-3,
            abs(rr[argmin(abs.(t .- 0.9))]/rr[1]/ana[argmin(abs.(t .- 0.9))] - 1),
            abs(rp[argmin(abs.(tp .- 0.9))]/cos2(p[argmin(abs.(tp .- 0.9)), 1])^-3 - 1))
end
axislegend(axa, position=:lb); axislegend(axb, position=:lt)
save(out, fig); println("wrote ", out)
