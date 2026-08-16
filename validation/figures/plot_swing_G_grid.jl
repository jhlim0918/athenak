# Six-run linear swing grid (doc fig, sec. 13): uniform vs moving-ring AMR at
# 4piG = 1.9 / 2.1 / 3.0 (amp = 1e-4), each against the linearized shearing-sheet
# ODE (continuum k^2, RK4). Data: validation/run/{sw2a_unif_mg, sw2c_amr,
# lin_unif_g21, lin_amr_g21, lin_unif_g30, lin_amr_g30}.user.hst.
using CairoMakie, Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13))
const HERE = @__DIR__
const RUN = joinpath(dirname(HERE), "run")
const A = 1e-4

function read_hst(f)
    rows = Float64[]; ncol = 0
    for ln in eachline(f)
        startswith(strip(ln), "#") && continue
        v = parse.(Float64, split(ln)); ncol = length(v); append!(rows, v)
    end
    permutedims(reshape(rows, ncol, :))
end

function ode(fourpig; kx0=-6.0, ky=1.0, q=1.5, Om=1.0, cs2=1.0, tend=8.0, dt=1e-4)
    u = (1.0, 0.0, 0.0); t = 0.0
    ts = Float64[]; ds = Float64[]
    f(t, u) = begin
        kx = kx0 + q*Om*ky*t
        c = cs2 - fourpig/(kx^2 + ky^2)
        (-(kx*u[2] + ky*u[3]), 2Om*u[3] + kx*c*u[1], -(2-q)*Om*u[2] + ky*c*u[1])
    end
    while t < tend
        push!(ts, t); push!(ds, u[1])
        k1 = f(t, u); k2 = f(t+dt/2, u .+ (dt/2).*k1)
        k3 = f(t+dt/2, u .+ (dt/2).*k2); k4 = f(t+dt, u .+ dt.*k3)
        u = u .+ (dt/6).*(k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        t += dt
    end
    ts[1:100:end], ds[1:100:end]
end

grid = [(1.9, "sw2a_unif_mg", "sw2c_amr"),
        (2.1, "lin_unif_g21", "lin_amr_g21"),
        (3.0, "lin_unif_g30", "lin_amr_g30")]

fig = Figure(size=(900, 900))
for (p, (fpg, bu, ba)) in enumerate(grid)
    hu = read_hst(joinpath(RUN, bu * ".user.hst"))
    ha = read_hst(joinpath(RUN, ba * ".user.hst"))
    tt, dd = ode(fpg)
    ax = Axis(fig[p, 1]; ylabel="d_cos / A", xlabel=p == 3 ? "t" : "",
              title=@sprintf("4πG = %.1f", fpg))
    lines!(ax, hu[:, 1], hu[:, 3] ./ A; color=:steelblue, linewidth=2.4, label="uniform")
    lines!(ax, ha[:, 1], ha[:, 3] ./ A; color=:crimson, linewidth=1.2, label="AMR (moving ring)")
    lines!(ax, tt, dd; color=:black, linewidth=1.0, linestyle=:dash, label="linear theory")
    iu = argmax(abs.(hu[:, 3])); ia = argmax(abs.(ha[:, 3]))
    @printf("4πG=%.1f: uniform peak %+.4f, AMR %+.4f, theory %+.4f ; max|unif-AMR|/A = %.2e\n",
            fpg, hu[iu, 3]/A, ha[ia, 3]/A, dd[argmax(abs.(dd))],
            maximum(abs.(hu[1:min(end, size(ha, 1)), 3] .-
                         ha[1:min(size(hu, 1), end), 3])) / A)
    p == 1 && axislegend(ax; position=:lb, framevisible=false, labelsize=11)
end
Label(fig[0, 1], "linear swing (amp = 10⁻⁴): uniform vs AMR vs theory across the Jeans threshold";
      fontsize=15)
save(joinpath(HERE, "swing_G_grid.png"), fig)
println("wrote figures/swing_G_grid.png")
fig
