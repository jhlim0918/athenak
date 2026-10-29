# Figures for fargo_amr_validation.pdf, nonlinear swing to clump collapse with orbital
# advection on refined meshes (4 pi G = 6, amp 0.2, Jeans criterion lambda_J >= 16 cells,
# up to 3 levels):
#   fargo_amr_clump.png       -- hydro: peak density, winding-wave amplitude d_cos/A,
#                                MeshBlock count, for uniform 128^2 / 256^2, rings, patches
#   fargo_amr_clump_maps.png  -- hydro: midplane density at t = 10 with MeshBlock outlines,
#                                full box and a zoom on the clump
#   fargo_amr_clump_mhd.png   -- MHD (B0 along y): peak density, d_cos/A, div B and the
#                                shared-face mismatch, MeshBlock count
# Run directories: FARGO_AMR_RUNS (default validation/run/fargo_amr), subdirectory clump/
# (H* hydro, M* MHD, each with SwingSG.user.hst, rhomax.txt from rhomax.py, maps/ from
# maps.py); the ring-era runs of multigrid_selfgravity_validation.pdf in clump/old/.
# Run from validation/: julia figures/plot_fargo_amr_clump.jl
using CairoMakie
using Printf
legend!(ax; kw...) = (try axislegend(ax; kw...) catch end)
CairoMakie.activate!(type="png")
set_theme!(Theme(fontsize=13, Axis=(xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25))))
const C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e44ad"

const HERE = @__DIR__
const VAL = dirname(HERE)
const RUNS = joinpath(get(ENV, "FARGO_AMR_RUNS", joinpath(VAL, "run", "fargo_amr")), "clump")
const AMP = 0.2
# lambda_J = 2 pi c_s / sqrt(4 pi G rho) = 52.1 root cells at rho = 1 (4 pi G = 6, 128^2)
const NJ0 = 2pi / sqrt(6.0) / (2pi / 128)
rho_at(ncell, level) = (NJ0 * 2.0^level / ncell)^2      # density where lambda_J = ncell cells

function readnum(f)
    rows = Float64[]; ncol = 0
    for ln in eachline(f)
        s = strip(ln)
        (isempty(s) || startswith(s, "#")) && continue
        v = parse.(Float64, split(s)); ncol = length(v); append!(rows, v)
    end
    permutedims(reshape(rows, ncol, :))
end
have(d, f) = isfile(joinpath(RUNS, d, f))
hst(d) = readnum(joinpath(RUNS, d, "SwingSG.user.hst"))
rmx(d) = readnum(joinpath(RUNS, d, "rhomax.txt"))

hydro = [("H1_uniform", "uniform 128²",              C1, :solid),
         ("H4_uni256",  "uniform 256²",              :black, :solid),
         ("H2_ring",    "AMR, x₂ rings",             C2, :solid),
         ("H3_patch",   "AMR, patches",              C3, :solid),
         ("H5_marginal", "AMR patches, 4πG=5, amp 0.05", C5, :solid)]

# ---------------------------------------------------------------- hydro histories
fig = Figure(size=(1200, 500))
ax1 = Axis(fig[1, 1], xlabel="t Ω", ylabel="max ρ / ρ₀", yscale=log10,
           title="peak density")
ax2 = Axis(fig[1, 2], xlabel="t Ω", ylabel="d_cos / A", title="winding wave")
ax3 = Axis(fig[1, 3], xlabel="t Ω", ylabel="MeshBlocks", yscale=log10,
           title="MeshBlocks (16³ cells)")
for L in 0:3
    hlines!(ax1, [rho_at(16, L)], color=(:gray, 0.6), linestyle=:dash, linewidth=1)
    lab = L == 2 ? "λ_J = 16 cells at level 2 = 4 cells at 128² (Truelove)" :
                   "λ_J = 16 cells at level $L"
    text!(ax1, 0.2, rho_at(16, L) * 1.08, text=lab, color=:gray35, fontsize=10)
end
old = joinpath(RUNS, "old", "clump_amr.rhomax.txt")
if isfile(old)
    m = readnum(old)
    lines!(ax1, m[:, 1], m[:, 3], color=(:gray40, 0.9), linestyle=:dash, linewidth=1.5,
           label="ring-era AMR (density criterion, 1 level)")
end
for (d, lab, col, ls) in hydro
    if have(d, "rhomax.txt")
        m = rmx(d)
        lines!(ax1, m[:, 1], m[:, 2], color=col, linestyle=ls, linewidth=2, label=lab)
        if !(d in ("H1_uniform", "H4_uni256", "H5_marginal"))      # marginal: never refines
            lines!(ax3, m[:, 1], m[:, 3], color=col, linewidth=2, label=lab * " (all)")
            lines!(ax3, m[:, 1], max.(m[:, 7], 0.8), color=col, linestyle=:dot, linewidth=1.5,
                   label=lab * " (level 3)")
        end
    end
    if have(d, "SwingSG.user.hst") && d != "H5_marginal"
        h = hst(d)
        lines!(ax2, h[:, 1], h[:, 3] ./ AMP, color=col, linestyle=ls, linewidth=2, label=lab)
    end
end
ylims!(ax3, 30, nothing)
ylims!(ax1, 0.7, 2.5e3)
try
    Legend(fig[2, 1:2], ax1, orientation=:horizontal, nbanks=2, framevisible=false,
           labelsize=11)
catch
end
legend!(ax3, position=:lt, framevisible=false, labelsize=10)
save(joinpath(HERE, "fargo_amr_clump.png"), fig)
println("wrote figures/fargo_amr_clump.png")

# ---------------------------------------------------------------- hydro maps
function readmap(d, dump)
    t = joinpath(RUNS, d, "maps", @sprintf("map_%05d.txt", dump))
    isfile(t) || return nothing
    hdr = split(strip(first(eachline(t)), ['#', ' ']))
    time = parse(Float64, hdr[1]); N = parse(Int, hdr[2])
    ext = parse.(Float64, hdr[3:6])
    img = Array{Float32}(undef, N, N)            # file is [j][i] row-major = column-major [i, j]
    read!(joinpath(RUNS, d, "maps", @sprintf("map_%05d.f32", dump)), img)
    blk = readnum(t)
    (; time, N, ext, img, blk)
end
const LEVCOL = Dict(0 => (:gray60, 0.5), 1 => ("#d62728", 0.9), 2 => ("#e377c2", 1.0),
                    3 => (:black, 0.45))
function outline!(ax, mp; zoom=nothing, yshift=0.0, ylen=0.0)
    # level relative to the run's own root grid (the coarsest block present)
    lmax = round(Int, log2(maximum(mp.blk[:, 5])))
    for r in eachrow(mp.blk)
        L = lmax - round(Int, log2(r[5]))
        x0, x1, y0, y1 = r[1:4]
        if yshift != 0.0          # image rolled in y: wrap the block into the shown window
            ymin = mp.ext[3] + yshift
            while y0 < ymin - 1e-9; y0 += ylen; y1 += ylen; end
            while y0 >= ymin + ylen - 1e-9; y0 -= ylen; y1 -= ylen; end
        end
        if zoom !== nothing
            (x1 < zoom[1] || x0 > zoom[2] || y1 < zoom[3] || y0 > zoom[4]) && continue
        end
        lines!(ax, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=LEVCOL[L],
               linewidth=(L == 0 ? 0.4 : 0.7))
    end
end
# maprows: (run, label) shown at dump, or (run, label, dump) with a per-run dump
function mapfig(maprows, dump, fname; cr=(-1.0, 3.0))
mps = [(r[1], r[2], readmap(r[1], length(r) > 2 ? r[3] : dump)) for r in maprows]
mps = filter(x -> x[3] !== nothing, mps)
if !isempty(mps)
    # zoom window: around each run's densest pixel; the image is periodic in y, so the
    # zoom is drawn from a copy shifted to put the peak at the centre
    hw = 0.6
    fig = Figure(size=(300 * length(mps) + 160, 680))
    local hm
    for (n, (d, lab, mp)) in enumerate(mps)
        xs = range(mp.ext[1], mp.ext[2], length=mp.N + 1)
        ys = range(mp.ext[3], mp.ext[4], length=mp.N + 1)
        lr = log10.(max.(mp.img, 1f-3))
        k = argmax(mp.img); dxp = (mp.ext[2] - mp.ext[1]) / mp.N
        xc = mp.ext[1] + (k[1] - 0.5) * dxp; yc = mp.ext[3] + (k[2] - 0.5) * dxp
        xc = clamp(xc, mp.ext[1] + hw, mp.ext[2] - hw)     # x is not periodic in the image
        zoom = (xc - hw, xc + hw, yc - hw, yc + hw)
        sh = mp.N ÷ 2 - k[2]                        # roll in y: peak row to the middle
        lz = circshift(lr, (0, sh)); ysh = ys .- sh * dxp
        a = Axis(fig[1, n], aspect=DataAspect(), title=@sprintf("%s, t = %.1f", lab, mp.time),
                 xlabel="x / H", ylabel=(n == 1 ? "y / H" : ""),
                 xgridvisible=false, ygridvisible=false)
        hm = heatmap!(a, xs, ys, lr, colormap=:cividis, colorrange=cr)
        outline!(a, mp)
        lines!(a, [zoom[1], zoom[2], zoom[2], zoom[1], zoom[1]],
               [zoom[3], zoom[3], zoom[4], zoom[4], zoom[3]], color=C2, linewidth=1.5)
        limits!(a, mp.ext[1], mp.ext[2], mp.ext[3], mp.ext[4])
        b = Axis(fig[2, n], aspect=DataAspect(),
                 title=@sprintf("zoom: max ρ = %.0f ρ₀", maximum(mp.img)),
                 xlabel="x / H", ylabel=(n == 1 ? "y / H" : ""),
                 xgridvisible=false, ygridvisible=false)
        heatmap!(b, xs, ysh, lz, colormap=:cividis, colorrange=cr)
        outline!(b, mp; zoom=zoom, yshift=-sh * dxp, ylen=mp.ext[4] - mp.ext[3])
        limits!(b, zoom...)
    end
    Colorbar(fig[1:2, length(mps) + 1], hm, label="log₁₀ ρ / ρ₀ (midplane)")
    Label(fig[3, 1:length(mps)], "MeshBlock outlines: gray level 0, red level 1, pink level 2, black level 3",
          fontsize=11, color=:gray25)
    save(joinpath(HERE, fname), fig)
    println("wrote figures/", fname)
end
end
mapfig([("H1_uniform", "uniform 128²"), ("H4_uni256", "uniform 256²"),
        ("H2_ring", "AMR, x₂ rings"), ("H3_patch", "AMR, patches")], 200,
       "fargo_amr_clump_maps.png")

# ---------------------------------------------------------------- MHD
mhd = [("M1_uni_b0.5",    "uniform 128², B₀ = 0.5",  C1, :solid),
       ("M2_ring_b0.5",   "rings, B₀ = 0.5",         C2, :solid),
       ("M3_patch_b0.5",  "patches, B₀ = 0.5",       C3, :solid),
       ("M4_uni_b0.25",   "uniform 128², B₀ = 0.25", C1, :dash),
       ("M5_patch_b0.25", "patches, B₀ = 0.25",      C3, :dash),
       ("M6_uni256_b0.5", "uniform 256², B₀ = 0.5",  :black, :solid)]
if any(have(d, "SwingSG.user.hst") for (d, _, _, _) in mhd)
    fig = Figure(size=(1200, 760))
    a1 = Axis(fig[1, 1], xlabel="t Ω", ylabel="max ρ / ρ₀", yscale=log10, title="peak density")
    a2 = Axis(fig[1, 2], xlabel="t Ω", ylabel="d_cos / A", title="winding wave")
    a3 = Axis(fig[2, 1], xlabel="t Ω", ylabel="ratio", yscale=log10,
              title="∫|∇·B|Δx dV / ∫|B| dV (lines), shared-face mismatch (dots)")
    a4 = Axis(fig[2, 2], xlabel="t Ω", ylabel="MeshBlocks", yscale=log10, title="MeshBlocks")
    for (d, lab) in [("H1_uniform", "hydro uniform 128²"), ("H3_patch", "hydro patches")]
        have(d, "rhomax.txt") || continue
        m = rmx(d)
        lines!(a1, m[:, 1], m[:, 2], color=(:gray50, 0.8), linewidth=1.2,
               linestyle=(d == "H1_uniform" ? :dot : :solid), label=lab)
    end
    for (d, lab, col, ls) in mhd
        if have(d, "rhomax.txt")
            m = rmx(d)
            lines!(a1, m[:, 1], m[:, 2], color=col, linestyle=ls, linewidth=2, label=lab)
            occursin("uni", d) || lines!(a4, m[:, 1], m[:, 3], color=col, linestyle=ls,
                                         linewidth=2, label=lab)
        end
        if have(d, "SwingSG.user.hst")
            h = hst(d)
            lines!(a2, h[:, 1], h[:, 3] ./ AMP, color=col, linestyle=ls, linewidth=2, label=lab)
            if size(h, 2) >= 14        # 1-based: 10 divb, 11 babs, 13 bmis, 14 bnrm
                dv = h[:, 10] ./ max.(h[:, 11], 1e-300)
                lines!(a3, h[:, 1], max.(dv, 1e-18), color=col, linestyle=ls, linewidth=2,
                       label=lab)
                bm = h[:, 13] ./ max.(h[:, 14], 1e-300)
                ok = h[:, 14] .> 0
                any(ok) && scatter!(a3, h[ok, 1], max.(bm[ok], 1e-18), color=col, markersize=4,
                                   marker=(ls == :solid ? :circle : :utriangle))
            end
        end
    end
    for (v, lab, c) in [(rho_at(4, 0), "Truelove (4 cells), 128²", :crimson),
                        (rho_at(16, 3), "λ_J = 16 cells at level 3", :gray35),
                        (rho_at(4, 3), "Truelove (4 cells) at level 3", :crimson)]
        hlines!(a1, [v], color=(c, 0.5), linestyle=:dot, linewidth=1.5)
        text!(a1, 5.2, v * 1.3, text=lab, color=c, fontsize=10)
    end
    legend!(a1, position=:lt, framevisible=false, labelsize=10)
    legend!(a4, position=:lt, framevisible=false, labelsize=10)
    save(joinpath(HERE, "fargo_amr_clump_mhd.png"), fig)
    println("wrote figures/fargo_amr_clump_mhd.png")
end
# B0 = 0.5, each refined run and the 256^2 run at its collapse (rho ~ 400), the 128^2 run at t = 10
mapfig([("M1_uni_b0.5", "uniform 128²", 200), ("M6_uni256_b0.5", "uniform 256²", 220),
        ("M2_ring_b0.5", "rings", 196), ("M3_patch_b0.5", "patches", 196)], 196,
       "fargo_amr_clump_mhd_maps.png")
