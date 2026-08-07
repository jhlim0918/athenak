# Reader for AthenaK .bin output files (file_type=bin), ported from
# vis/python/bin_convert.py read_binary(). Unlike legacy VTK output, the bin
# format stores per-MeshBlock data with logical locations and levels, so it is
# correct for SMR/AMR runs.
#
# Usage: include this file, then
#   fd = read_bin("path/to/file.bin")
#   img, x1f, x2f = assemble_root_slice(fd, "dens")   # z≈0 slice on root grid
#
# fd fields: time, cycle, var_names, Nx1/2/3 (root cells), x1min...x3max,
#   nx_mb (cells per block), n_mbs, mb_logical (n_mbs x 4: lx1,lx2,lx3,level),
#   mb_geometry (n_mbs x 6: block x1min,x1max,x2min,x2max,x3min,x3max),
#   mb_data: Dict(var => Vector of (nx1,nx2,nx3) arrays)

struct BinFileData
    time::Float64
    cycle::Int
    var_names::Vector{String}
    Nx1::Int; Nx2::Int; Nx3::Int
    x1min::Float64; x1max::Float64
    x2min::Float64; x2max::Float64
    x3min::Float64; x3max::Float64
    nx_mb::NTuple{3,Int}
    n_mbs::Int
    mb_logical::Matrix{Int}      # n_mbs x 4
    mb_geometry::Matrix{Float64} # n_mbs x 6
    mb_data::Dict{String,Vector{Array{Float64,3}}}
    header::Vector{String}
end

function _get_param(header::Vector{String}, block::String, key::String)
    blk = "<none>"
    for line in header
        if startswith(line, "<")
            blk = line
            continue
        end
        occursin("=", line) || continue
        k, v = strip.(split(line, "=", limit=2))
        if blk == block && k == key
            return String(v)
        end
    end
    error("no parameter $block/$key in bin header")
end

function read_bin(filename::AbstractString)
    io = open(filename, "r")
    code_header = split(readline(io))
    (length(code_header) >= 1 && code_header[1] == "Athena") ||
        error("bad file format (expected 'Athena' header)")
    version = split(code_header[end], "=")[end]
    version == "1.1" || error("unsupported bin format version $version")

    pheader_count = parse(Int, split(readline(io), "=")[end])
    pheader = Dict{String,String}()
    for _ in 1:(pheader_count-1)
        k, v = strip.(split(readline(io), "=", limit=2))
        pheader[k] = v
    end
    time = parse(Float64, pheader["time"])
    cycle = parse(Int, pheader["cycle"])
    locsize = parse(Int, pheader["size of location"])
    varsize = parse(Int, pheader["size of variable"])
    locT = locsize == 8 ? Float64 : Float32
    varT = varsize == 8 ? Float64 : Float32

    nvars = parse(Int, split(readline(io), "=")[end])
    var_names = String.(split(readline(io))[2:end])
    header_size = parse(Int, split(readline(io), "=")[end])
    raw_header = String(read(io, header_size))
    header = String[]
    for line in split(raw_header, "\n")
        line = strip(split(line, "#")[1])
        isempty(line) || push!(header, String(line))
    end

    Nx1 = parse(Int, _get_param(header, "<mesh>", "nx1"))
    Nx2 = parse(Int, _get_param(header, "<mesh>", "nx2"))
    Nx3 = parse(Int, _get_param(header, "<mesh>", "nx3"))
    nghost = parse(Int, _get_param(header, "<mesh>", "nghost"))
    x1min = parse(Float64, _get_param(header, "<mesh>", "x1min"))
    x1max = parse(Float64, _get_param(header, "<mesh>", "x1max"))
    x2min = parse(Float64, _get_param(header, "<mesh>", "x2min"))
    x2max = parse(Float64, _get_param(header, "<mesh>", "x2max"))
    x3min = parse(Float64, _get_param(header, "<mesh>", "x3min"))
    x3max = parse(Float64, _get_param(header, "<mesh>", "x3max"))

    mb_logical = Int[]
    mb_geometry = Float64[]
    mb_data = Dict{String,Vector{Array{Float64,3}}}(v => Array{Float64,3}[] for v in var_names)
    n_mbs = 0
    nx_mb = (0, 0, 0)
    while !eof(io)
        idx = [read(io, Int32) for _ in 1:6] .- Int32(nghost)
        n1 = idx[2] - idx[1] + 1
        n2 = idx[4] - idx[3] + 1
        n3 = idx[6] - idx[5] + 1
        nx_mb = (Int(n1), Int(n2), Int(n3))
        append!(mb_logical, Int.([read(io, Int32) for _ in 1:4]))
        append!(mb_geometry, Float64.([read(io, locT) for _ in 1:6]))
        # data stream is C-ordered [var][k][j][i], i fastest — matches Julia
        # column-major (n1,n2,n3,nvars)
        buf = Array{varT,4}(undef, n1, n2, n3, nvars)
        read!(io, buf)
        for (vi, v) in enumerate(var_names)
            push!(mb_data[v], Float64.(buf[:, :, :, vi]))
        end
        n_mbs += 1
    end
    close(io)

    return BinFileData(time, cycle, var_names, Nx1, Nx2, Nx3,
                       x1min, x1max, x2min, x2max, x3min, x3max,
                       nx_mb, n_mbs,
                       permutedims(reshape(mb_logical, 4, n_mbs)),
                       permutedims(reshape(mb_geometry, 6, n_mbs)),
                       mb_data, header)
end

# Restrict a fine array by 2x averaging in every dimension
function _restrict2(a::Array{Float64,3})
    return 0.125 .* (a[1:2:end,1:2:end,1:2:end] .+ a[2:2:end,1:2:end,1:2:end] .+
                     a[1:2:end,2:2:end,1:2:end] .+ a[2:2:end,2:2:end,1:2:end] .+
                     a[1:2:end,1:2:end,2:2:end] .+ a[2:2:end,1:2:end,2:2:end] .+
                     a[1:2:end,2:2:end,2:2:end] .+ a[2:2:end,2:2:end,2:2:end])
end

# Assemble the full root-level grid (Nx1,Nx2,Nx3): root blocks placed directly,
# level-L blocks restricted by L applications of conservative 2x averaging.
function assemble_root(fd::BinFileData, var::AbstractString)
    out = fill(NaN, fd.Nx1, fd.Nx2, fd.Nx3)
    m1, m2, m3 = fd.nx_mb
    for m in 1:fd.n_mbs
        lx1, lx2, lx3, lev = fd.mb_logical[m, :]
        r = fd.mb_data[var][m]
        for _ in 1:lev
            r = _restrict2(r)
        end
        s = 1 << lev
        i0, j0, k0 = lx1*m1÷s, lx2*m2÷s, lx3*m3÷s
        out[i0+1:i0+m1÷s, j0+1:j0+m2÷s, k0+1:k0+m3÷s] = r
    end
    any(isnan, out) && error("gaps in assembled root grid")
    return out
end

# Convenience: z≈0 slice (middle z index) of the assembled root grid, plus
# cell-edge coordinate ranges for plotting with heatmap!.
function assemble_root_slice(fd::BinFileData, var::AbstractString)
    a = assemble_root(fd, var)
    kz = fd.Nx3 ÷ 2 + 1
    x1edges = range(fd.x1min, fd.x1max, length=fd.Nx1+1)
    x2edges = range(fd.x2min, fd.x2max, length=fd.Nx2+1)
    return a[:, :, kz], x1edges, x2edges
end

# Block outlines for overlay plotting: returns vector of (rect, level) where
# rect = (x1min, x1max, x2min, x2max), one entry per unique block footprint
# (all z-columns share the same x1-x2 footprint in these setups).
function block_outlines(fd::BinFileData)
    seen = Set{NTuple{5,Float64}}()
    out = Tuple{NTuple{4,Float64},Int}[]
    for m in 1:fd.n_mbs
        g = fd.mb_geometry[m, :]
        lev = fd.mb_logical[m, 4]
        key = (g[1], g[2], g[3], g[4], Float64(lev))
        key in seen && continue
        push!(seen, key)
        push!(out, ((g[1], g[2], g[3], g[4]), Int(lev)))
    end
    return out
end
