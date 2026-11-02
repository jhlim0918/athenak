# Movie: two-mode swing collapse, uniform 256^2 against AMR (64^2 root, three levels),
# maximum density along z (log10), every 0.1 from t = 0 to 10, with the MeshBlock outlines
# of the AMR run (z = 0 layer) and the peak density against time.
# Frames: validation/run/fargo_amr/clump2m/<run>/maps/zmax_DDDDD.{f32,txt} from census.py.
# Run from validation/: julia figures/movie_clump2m.jl [RUN_A RUN_B LABEL_A LABEL_B OUT TITLE]
using CairoMakie
using Printf
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=14))

const HERE = @__DIR__
const DIR = joinpath(dirname(HERE), "run", "fargo_amr", "clump2m")
runA = get(ARGS, 1, "H_uni256"); runB = get(ARGS, 2, "H_amr")
labA = get(ARGS, 3, "uniform 256²"); labB = get(ARGS, 4, "AMR, 64² root, 3 levels (finest 256²)")
out = get(ARGS, 5, joinpath(DIR, "clump2m_hydro_uni_vs_amr.mp4"))
ttlbase = get(ARGS, 6, "two-mode swing collapse (hydro)")
const NOUT = 256          # frames resampled to the finest grid

function frame(run, n)
    t = joinpath(DIR, run, "maps", @sprintf("zmax_%05d.txt", n))
    hdr = split(strip(first(eachline(t)), ['#', ' ']))
    time = parse(Float64, hdr[1]); N = parse(Int, hdr[2]); ext = parse.(Float64, hdr[3:6])
    img = Array{Float32}(undef, N, N)
    read!(joinpath(DIR, run, "maps", @sprintf("zmax_%05d.f32", n)), img)
    r = NOUT ÷ N
    img = r > 1 ? repeat(img, inner=(r, r)) : img
    blk = Float64[]
    lines = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), readlines(t))
    rows = [parse.(Float64, split(l)) for l in lines]
    (; time, N, ext, img, rows)
end
# outline polylines per level (NaN-separated); level from pixels per cell r: N/64 = 2^lmax
function outlines(fr)
    pts = [Point2f[] for _ in 1:3]
    lmax = round(Int, log2(fr.N / 64))
    for r in fr.rows
        L = lmax - round(Int, log2(r[5]))
        (L < 0 || L > 2) && continue
        x0, x1, y0, y1 = r[1:4]
        append!(pts[L+1], Point2f[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0), (NaN, NaN)])
    end
    pts
end
peak(run) = begin
    v = Tuple{Float64,Float64}[]
    for l in eachline(joinpath(DIR, run, "census.txt"))
        m = match(r"t=([0-9.]+) peak3D ([0-9.e+]+)", l)
        m === nothing || push!(v, (parse(Float64, m[1]), parse(Float64, m[2])))
    end
    v
end
pA, pB = peak(runA), peak(runB)

fA, fB = frame(runA, 0), frame(runB, 0)
xs = range(fA.ext[1], fA.ext[2], length=NOUT + 1); ys = range(fA.ext[3], fA.ext[4], length=NOUT + 1)
imA = Observable(log10.(max.(fA.img, 1f-2))); imB = Observable(log10.(max.(fB.img, 1f-2)))
ol = [Observable(p) for p in outlines(fB)]
ttl = Observable("t = 0.0")
tline = Observable(0.0)

fig = Figure(size=(1180, 860))
Label(fig[0, 1:3], ttl, fontsize=20, font=:bold)
cr = (-0.5, 3.0)
axA = Axis(fig[1, 1], aspect=DataAspect(), title=labA, xlabel="x / H", ylabel="y / H")
axB = Axis(fig[1, 2], aspect=DataAspect(), title=labB, xlabel="x / H")
hm = heatmap!(axA, xs, ys, imA, colormap=:cividis, colorrange=cr)
heatmap!(axB, xs, ys, imB, colormap=:cividis, colorrange=cr)
for (L, c, w) in zip(0:2, [(:gray70, 0.5), ("#d62728", 0.8), ("#e377c2", 0.8)], [0.4, 0.6, 0.6])
    lines!(axB, ol[L+1], color=c, linewidth=w)
end
for a in (axA, axB)
    limits!(a, fA.ext...)
    hidedecorations!(a, label=false, ticklabels=false, ticks=false)
end
Colorbar(fig[1, 3], hm, label="log₁₀ max_z ρ / ρ₀")
axP = Axis(fig[2, 1:2], xlabel="t Ω", ylabel="max ρ / ρ₀", yscale=log10, height=200)
lines!(axP, first.(pA), last.(pA), color="#2a78d6", linewidth=2, label=labA)
lines!(axP, first.(pB), last.(pB), color="#eb6834", linewidth=2, label=labB)
hlines!(axP, [683.0], color=(:crimson, 0.6), linestyle=:dot)
text!(axP, 0.2, 800.0, text="Truelove limit at 256² (λ_J = 4 cells)", color=:crimson, fontsize=11)
vlines!(axP, tline, color=:black, linewidth=1)
axislegend(axP, position=:lt, framevisible=false, labelsize=11)
Label(fig[3, 1:2], "AMR outlines (z = 0 layer): gray level 0, red level 1, pink level 2",
      fontsize=11, color=:gray30)

record(fig, out, 0:100; framerate=8) do n
    a, b = frame(runA, n), frame(runB, n)
    imA[] = log10.(max.(a.img, 1f-2)); imB[] = log10.(max.(b.img, 1f-2))
    for (o, p) in zip(ol, outlines(b)); o[] = p; end
    ttl[] = @sprintf("%s, t = %.1f", ttlbase, a.time)
    tline[] = a.time
end
println("wrote ", out)
