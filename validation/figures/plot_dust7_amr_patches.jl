# dust7 test 5: AMR (one level, rho_p > 50) of the Godines+26 10 au 64^3 box from the same
# 20 P state with and without orbital advection: face-on Sigma_p/Sigma_g0 at 20.5 P with the
# refined MeshBlocks outlined (projected on x-y; with orbital advection they are full rings).
#   julia plot_dust7_amr_patches.jl <prefix_fargo> <prefix_nofargo> out.png
using CairoMakie, Printf
p1, p2, out = ARGS
function load(p)
    h = split(readline(p * ".txt")); t = parse(Float64, h[1]); nf = parse(Int, h[2])
    x0 = parse(Float64, h[3]); x1 = parse(Float64, h[4])
    S = reshape(reinterpret(Float64, read(p * ".map")), nf, nf)
    B = [parse.(Float64, split(l)) for l in eachline(p * ".blocks")]
    t, range(x0 + (x1-x0)/(2nf), x1 - (x1-x0)/(2nf), length=nf), S, B
end
D = [load(p1), load(p2)]
hi = maximum(maximum(d[3]) for d in D)
fig = Figure(size=(1250, 620), fontsize=15)
local hm
for (n, (lab, (t, x, S, B))) in enumerate(zip(("orbital advection: refined rings", "no orbital advection: refined patches"), D))
    nb = length(unique([(b[1], b[3]) for b in B]))
    ax = Axis(fig[1, n], aspect=DataAspect(), xlabel=L"x/H", ylabel=(n == 1 ? L"y/H" : ""),
              title=@sprintf("%s\nt = %.1f P, %d refined blocks (%d columns in x-y)", lab, t, length(B), nb))
    global hm = heatmap!(ax, x, x, S, colormap=:cividis, colorscale=log10, colorrange=(1e-2, hi))
    for b in unique([(b[1], b[2], b[3], b[4]) for b in B])
        lines!(ax, [b[1], b[2], b[2], b[1], b[1]], [b[3], b[3], b[4], b[4], b[3]], color=:magenta, linewidth=1)
    end
end
Colorbar(fig[1, 3], hm, label=L"\Sigma_p/\Sigma_{g,0}")
save(out, fig); println("wrote ", out)
