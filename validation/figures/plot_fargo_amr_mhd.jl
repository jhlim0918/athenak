# Figure for fargo_amr_validation.pdf, MHD on rings (branch fargo-amr-mhd):
#   left  -- c_y (azimuthal field perturbation) of the refined-MHD reproducer
#            (inputs/tests/swing_mhd_repro.athinput: B0 = 1, no gravity, static ring
#            |x| < pi/4, 64^2 x 4 root) against linear theory: the ring without orbital
#            advection (the known field error), the ring with it, and the uniform twins;
#   right -- mismatch of the two copies of the faces shared by fine and coarse blocks,
#            Sum|Bn(coarse) - <Bn(fine)>| dA / Sum|Bn| dA (history bmis/bnrm), with and
#            without the EMF correction of the face-field remap, for the ring, a mesh with
#            x3 level jumps as well, and a moving ring (AMR).
# Run directories: FARGO_AMR_RUNS (default validation/run/fargo_amr), subdirectory mhd/.
# Run from validation/: julia figures/plot_fargo_amr_mhd.jl
using CairoMakie
using Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13, Axis=(xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25))))
const C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

const HERE = @__DIR__
const VAL = dirname(HERE)
const RUNS = joinpath(get(ENV, "FARGO_AMR_RUNS", joinpath(VAL, "run", "fargo_amr")), "mhd")

function read_hst(f)
    rows = Float64[]; ncol = 0
    for ln in eachline(f)
        startswith(strip(ln), "#") && continue
        v = parse.(Float64, split(ln)); ncol = length(v); append!(rows, v)
    end
    permutedims(reshape(rows, ncol, :))
end
hst(d) = read_hst(joinpath(RUNS, d, "SwingMHD.user.hst"))

# linear theory (validation/run/swing_mhd/swing_ode.py, no gravity): RK4 on the 5-ODE system
function theory(tt; kx0=-6.0, ky=1.0, b0=1.0, om=1.0, q=1.5, cs=1.0, amp=1e-4)
    f(t, y) = begin
        d, wx, wy, cx, cy = y
        kx = kx0 + q*om*ky*t
        [-(kx*wx + ky*wy), 2om*wy + kx*cs^2*d - b0*(ky*cx - kx*cy),
         -(2 - q)*om*wx + ky*cs^2*d, ky*b0*wx, -kx*b0*wx - q*om*cx]
    end
    y = [amp, 0, 0, 0, 0]; t = 0.0; h = 1e-3; out = zeros(length(tt))
    for (n, te) in enumerate(tt)
        while t < te - 1e-12
            hh = min(h, te - t)
            k1 = f(t, y); k2 = f(t + hh/2, y .+ hh/2 .* k1)
            k3 = f(t + hh/2, y .+ hh/2 .* k2); k4 = f(t + hh, y .+ hh .* k3)
            y = y .+ hh/6 .* (k1 .+ 2k2 .+ 2k3 .+ k4); t += hh
        end
        out[n] = y[5]
    end
    out
end

fig = Figure(size=(1100, 420))
ax = Axis(fig[1, 1], xlabel="t Ω", ylabel="c_y / A",
          title="reproducer: field perturbation, static ring")
tt = range(0, 6, length=601)
lines!(ax, tt, theory(tt) ./ 1e-4, color=:black, linewidth=2.5, label="linear theory")
for (d, lab, col, ls) in [("nf_ring", "ring, no orbital advection", C2, :solid),
                          ("nf_uni", "uniform, no orbital advection", C2, :dot),
                          ("f_ring", "ring, orbital advection", C1, :solid),
                          ("f_uni", "uniform, orbital advection", C1, :dot)]
    h = hst(d)
    lines!(ax, h[:, 1], h[:, 9] ./ 1e-4, color=col, linestyle=ls, linewidth=2, label=lab)
end
axislegend(ax, position=:lb, framevisible=false, labelsize=11)

ax2 = Axis(fig[1, 2], yscale=log10, xlabel="t Ω", ylabel="Σ|ΔBₙ| dA / Σ|Bₙ| dA",
           title="shared fine/coarse faces: copy mismatch")
for (d, lab, col, ls) in [("f_ring_nofc", "ring, no EMF correction", C1, :dash),
                          ("f_ring", "ring, EMF correction", C1, :solid),
                          ("z3_nofc", "x₁ + x₃ jumps, no correction", C3, :dash),
                          ("z3", "x₁ + x₃ jumps, correction", C3, :solid),
                          ("amr_nofc", "moving ring (AMR), no correction", C4, :dash),
                          ("amr", "moving ring (AMR), correction", C4, :solid)]
    h = hst(d)
    r = h[2:end, 13] ./ max.(h[2:end, 14], 1e-300)
    r[r .< 1e-17] .= 1e-17
    lines!(ax2, h[2:end, 1], r, color=col, linestyle=ls, linewidth=1.6, label=lab)
end
ylims!(ax2, 1e-17, 1e8)
axislegend(ax2, position=:rt, framevisible=false, labelsize=10, nbanks=2)
save(joinpath(HERE, "fargo_amr_mhd.png"), fig, px_per_unit=2)
println("wrote fargo_amr_mhd.png")
