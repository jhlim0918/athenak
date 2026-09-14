# Johansen & Youdin (2007) run BA, side by side at one time: the uniform 256^2 mesh (the
# Vista run, validation/run/dust6/BA_u256) and the four-level adaptive mesh on 8^2
# MeshBlocks (BA_amr4_mb8) with its MeshBlocks drawn.  Both at the same finest resolution;
# the particle counts in the panel titles are read from the phst files at that time (the
# adaptive run carries more: see the validation doc, sec. 12).
#   DUST6_RUN=<dir with the dust6 runs> julia plot_dust6_ba_sbs.jl [t]     (default t = 500)
using CairoMakie, Printf, Statistics, DelimitedFiles
CairoMakie.activate!(type="png"); set_theme!(Theme(fontsize=14))
const HERE = @__DIR__
const RUN  = get(ENV, "DUST6_RUN", joinpath(dirname(HERE), "run", "dust6"))
include(joinpath(dirname(HERE), "..", "scripts", "athenak_bin.jl"))
tsnap = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 500.0

# the dump of `base` under `dir` (top level or bin/) nearest to tsnap
function nearest_dump(dir, base, t)
    fs = String[]
    for d in (dir, joinpath(dir, "bin"))
        isdir(d) && append!(fs, joinpath.(d, filter(f -> occursin("$base.dust_dpm.", f), readdir(d))))
    end
    isempty(fs) && error("no $base.dust_dpm dumps under $dir")
    times = [read_bin(f).time for f in fs]
    fs[argmin(abs.(times .- t))]
end
# particle count of the run at time t, from its phst
function npart_at(phst, t)
    h = readdlm(phst, comments=true, comment_char='#'); k = argmin(abs.(h[:,1] .- t))
    Int(round(h[k,3]))
end
# the finest-level image of a (possibly refined) dust_dpm dump
function finest(fd)
    lev = fd.mb_logical[:,4]; lmax = maximum(lev); s = 1 << lmax
    a = fill(NaN, fd.Nx1*s, fd.Nx2*s); m1, m2 = fd.nx_mb[1], fd.nx_mb[2]
    for m in 1:fd.n_mbs
        lx1, lx2, _, l = fd.mb_logical[m,:]; b = fd.mb_data["dustdpm"][m][:,:,1]
        e = 1 << (lmax-l); e > 1 && (b = repeat(b, inner=(e,e)))
        i0, j0 = lx1*m1*e, lx2*m2*e; a[i0+1:i0+m1*e, j0+1:j0+m2*e] = b
    end
    (a, range(fd.x1min, fd.x1max, length=size(a,1)+1), range(fd.x2min, fd.x2max, length=size(a,2)+1),
     [count(lev .== l) for l in 0:lmax], fd.n_mbs*m1*m2, (fd.Nx1*s)*(fd.Nx2*s), lmax)
end

EPS0 = 0.2; CR = (-1.7, 0.8)
LC = (:white, :magenta, :springgreen, :red); LW = (2.0, 1.6, 1.0, 0.6)
panels = [(nearest_dump(joinpath(RUN, "BA_u256"), "si_BA", tsnap), joinpath(RUN, "BA_u256", "si_BA.phst"),
           "uniform 256²", false),
          (nearest_dump(joinpath(RUN, "BA_amr4_mb8"), "si_BA4m8", tsnap), joinpath(RUN, "BA_amr4_mb8", "si_BA4m8.phst"),
           "AMR, 4 levels, 8² MeshBlocks", true)]
fig = Figure(size=(1320, 700))
hm = nothing
for (n, (f, phst, lab, mesh)) in enumerate(panels)
    fd = read_bin(f); a, xe, ze, nlev, nc, nu, lmax = finest(fd)
    np = npart_at(phst, fd.time)
    ax = Axis(fig[1,n], aspect=DataAspect(), xlabel="x  (H)", ylabel=(n == 1 ? "z  (H)" : ""),
              title=@sprintf("%s: %.0f%% of the uniform cells, %.2g particles\nt = %.0f Ω⁻¹",
                             lab, 100*nc/nu, np, fd.time))
    global hm = heatmap!(ax, xe, ze, log10.(max.(a ./ EPS0, 1e-2)), colormap=:cividis, colorrange=CR)
    if mesh
        for (r, l) in block_outlines(fd)
            lines!(ax, [r[1],r[2],r[2],r[1],r[1]], [r[3],r[3],r[4],r[4],r[3]],
                   color=(LC[l+1], 0.95), linewidth=LW[l+1])
        end
        hs = []; ls = String[]
        for l in 0:lmax
            nlev[l+1] > 0 || continue
            push!(hs, LineElement(color=LC[l+1], linewidth=2.0))
            push!(ls, @sprintf("level %d: %d blocks (%d² eff.)", l, nlev[l+1], fd.Nx1 << l))
        end
        axislegend(ax, hs, ls, position=:lb, framevisible=true, framecolor=(:white, 0.4),
                   backgroundcolor=(:black, 0.55), labelcolor=:white, labelsize=11, patchsize=(24, 11))
    end
    n == 2 && hideydecorations!(ax, grid=false)
end
Colorbar(fig[1,3], hm, label="log₁₀ ρ_p / ⟨ρ_p⟩", width=14)
Label(fig[0,1:3], "Johansen & Youdin (2007) run BA, saturated state: uniform vs adaptive mesh",
      fontsize=17, font=:bold, tellwidth=false)
out = joinpath(HERE, "dust6_ba_sbs.png")
save(out, fig, px_per_unit=1.6)
println("wrote ", out)
