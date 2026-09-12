# Our BA run in the convention of Johansen & Youdin (2007) Fig. 2: particle density
# rho_p/rho_g on a LINEAR scale from 0 (black) to 1 (bright), axes in eta r.
using CairoMakie, Printf, Statistics
CairoMakie.activate!(type="png")
include("/private/tmp/claude-501/-Users-jaysmac-Library-CloudStorage-Dropbox-Research-code/a8f424f7-db89-4a52-b703-59ddfbd9dce2/scratchpad/wt-dustsmr/scripts/athenak_bin.jl")
const ETAR = 0.05
const EPS0 = 1.0      # AB: eps = 1 (BA: 0.2)
const CMAX = 5.0      # JY07 Fig. 5 scale for AB (Fig. 2 uses 1 for BA)
const TICKS = -1:0.5:1   # AB box is 2 eta r  (BA: -20:10:20)

function finest(fd)                   # dust density on the finest-level grid
    lmax = maximum(fd.mb_logical[:,4]); s = 1 << lmax
    a = fill(NaN, fd.Nx1*s, fd.Nx2*s); m1, m2 = fd.nx_mb[1], fd.nx_mb[2]
    for m in 1:fd.n_mbs
        lx1, lx2, _, lev = fd.mb_logical[m,:]; blk = fd.mb_data["dustdpm"][m][:,:,1]
        e = 1 << (lmax-lev); e > 1 && (blk = repeat(blk, inner=(e,e)))
        i0, j0 = lx1*m1*e, lx2*m2*e; a[i0+1:i0+m1*e, j0+1:j0+m2*e] = blk
    end
    a
end
function pick(dir, base, t)
    fs = sort(filter(f->occursin("$base.dust_dpm.", f), readdir(dir)))
    ts = [read_bin(joinpath(dir,f)).time for f in fs]
    read_bin(joinpath(dir, fs[argmin(abs.(ts .- t))]))
end
B  = "/private/tmp/claude-501/-Users-jaysmac-Library-CloudStorage-Dropbox-Research-code/ba03dd58-d645-460b-8f83-2bf1967494fe/scratchpad/run/AB_u256/bin"; C = B
panels = [(B,8.0), (B,16.0), (B,24.0), (B,32.0)]
cmap = cgrad([:black, RGBf(0.10,0.10,0.45), RGBf(0.35,0.15,0.60), RGBf(0.85,0.20,0.25),
              RGBf(1.0,0.55,0.0), RGBf(1.0,0.90,0.35), :white])
fig = Figure(size=(1050,1080), backgroundcolor=:white)
for (n,(dir,t)) in enumerate(panels)
    fd = pick(dir, "si_AB_u256", t); a = finest(fd)
    xr = range(fd.x1min/ETAR, fd.x1max/ETAR, length=size(a,1)+1)
    zr = range(fd.x2min/ETAR, fd.x2max/ETAR, length=size(a,2)+1)
    r, c = divrem(n-1, 2) .+ (1, 1)
    ax = Axis(fig[r,c], aspect=DataAspect(), title=@sprintf("t = %.1f  Ω⁻¹", fd.time),
              xlabel = r == 2 ? "x/(ηr)" : "", ylabel = c == 1 ? "z/(ηr)" : "",
              xticks=TICKS, yticks=TICKS)
    heatmap!(ax, xr, zr, a, colormap=cmap, colorrange=(0,CMAX))  # rho_p/rho_g, rho_g = 1
    # z-averaged radial profile: is there one dominant filament?
    px = vec(mean(a, dims=2))
    @printf("t=%5.1f  max rho_p/rho_g=%6.2f (=%4.1fx mean)  z-avg radial: max=%.2f at x=%.1f etar, contrast max/med=%.2f\n",
            fd.time, maximum(a), maximum(a)/EPS0, maximum(px),
            (fd.x1min + (argmax(px)-0.5)*(fd.x1max-fd.x1min)/length(px))/ETAR,
            maximum(px)/median(px))
end
Label(fig[0,1:2], "JY07 run AB in the convention of their Fig. 5 (ρ_p/ρ_g, linear 0→5)",
      fontsize=17, font=:bold)
save("/private/tmp/claude-501/-Users-jaysmac-Library-CloudStorage-Dropbox-Research-code/ba03dd58-d645-460b-8f83-2bf1967494fe/scratchpad/AB_jy07_fig5_style.png", fig, px_per_unit=1.5)
println("wrote /private/tmp/claude-501/-Users-jaysmac-Library-CloudStorage-Dropbox-Research-code/ba03dd58-d645-460b-8f83-2bf1967494fe/scratchpad/AB_jy07_fig5_style.png")
