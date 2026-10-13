# Vista GH200 scaling of the stratified SI box (streaming_strat, Godines+26 10 au setup,
# 1 particle per cell, 4 species), from the "# bench" lines of the si_strat_scaling scan
# (cycles 100-400 of each run.log):
#   (a) strong scaling: s/cycle vs GPUs at fixed size, ideal lines from the smallest count
#   (b) weak scaling at 16.8M cells (and particles) per GPU: cell-updates/s per GPU
#   julia plot_si3d_scaling.jl <scan_dir> out.png

using CairoMakie, Printf

scandir = ARGS[1]; out = ARGS[2]
pts = NamedTuple[]
for d in readdir(scandir, join=true)
    f = joinpath(d, "run.log"); isfile(f) || continue
    occursin(r"^N\d", basename(d)) || continue
    l = filter(s -> startswith(s, "# bench N="), readlines(f)); isempty(l) && continue
    b = l[end]
    n = parse.(Int, split(match(r"N=([0-9x]+)", b)[1], "x")); length(n) == 1 && (n = fill(n[1], 3))
    push!(pts, (cells = prod(n), lab = join(n, "×"), mb = parse(Int, match(r"MB=(\d+)", b)[1]),
                g = parse(Int, match(r"GPUs=(\d+)", b)[1]), spc = parse(Float64, match(r"([0-9.]+) s/cycle", b)[1]),
                pg = parse(Float64, match(r"\(([0-9.e+]+) per GPU\)", b)[1])))
end

fig = Figure(size=(1300, 520), fontsize=15)
axa = Axis(fig[1, 1], xscale=log2, yscale=log10, xlabel="GPUs (GH200, one per node)", ylabel="wall time per cycle [s]",
           title="(a) strong scaling (dashed: ideal; numbers: efficiency)", xticks=[1, 2, 4, 8, 16, 32, 64])
for (lab, mb, col) in (("256×256×256", 64, :dodgerblue), ("512×512×512", 128, :crimson))
    s = sort(filter(p -> p.lab == lab && p.mb == mb, pts), by = p -> p.g)
    g = [p.g for p in s]; t = [p.spc for p in s]
    scatterlines!(axa, g, t, color=col, label="$lab (MeshBlock $(mb)³)")
    lines!(axa, g, t[1]*g[1]./g, color=col, linestyle=:dash)
    for p in s
        text!(axa, p.g, p.spc, text=@sprintf("%.2f", t[1]*g[1]/(p.spc*p.g)), color=col,
              align=(:left, :bottom), offset=(4, 2), fontsize=11)
    end
end
axislegend(axa, position=:lb, labelsize=12)
xlims!(axa, 0.8, 45)

axb = Axis(fig[1, 2], xscale=log2, yscale=log10, xlabel="GPUs (GH200, one per node)",
           ylabel="cell-updates/s per GPU", title="(b) weak scaling, 16.8M cells + particles per GPU",
           xticks=[1, 2, 4, 8, 16, 32, 64])
w = sort(filter(p -> p.cells ÷ p.g == 256^3 && !(p.lab == "256×256×256" && p.mb == 128), pts), by = p -> p.g)
scatterlines!(axb, [p.g for p in w], [p.pg for p in w], color=:black)
for p in w
    text!(axb, p.g, p.pg, text="$(p.lab)\n" * @sprintf("%.2f", p.pg/w[1].pg), align=(:center, :bottom),
          offset=(0, 6), fontsize=11)
end
hlines!(axb, [w[1].pg], color=:gray, linestyle=:dash)
ylims!(axb, 1e7, 3e8); xlims!(axb, 0.6, 128)
Label(fig[2, 1:2], "AthenaK streaming_strat on Vista gh (CUDA, Kokkos unified memory, CUDA-unaware OpenMPI 5.0.5); " *
      "(0.2H)³ box for 256³/512³, (0.4H)²×0.2H and (0.8H)×(0.4H)×0.2H for the 32/64-GPU points; cycles 100-400.",
      fontsize=12, tellwidth=false)
save(out, fig); println("wrote ", out)
