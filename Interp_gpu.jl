module InterpGPU

using CUDA

export GPUGridInterpolator, gpu_linear_interpolator, interpolate, interpolate!

function interp_weights_kernel!(i_lo, w, x_grid, x_min, x_step, ngrid)
    n = Int32(length(x_grid))
    x_max = x_min + x_step * Float32(ngrid - Int32(1))
    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = gridDim().x * blockDim().x
    @inbounds begin
        while i <= n
            x = x_grid[i]
            if x <= x_min
                i_lo[i] = Int32(1)
                w[i] = 0.0f0
            elseif x >= x_max
                i_lo[i] = ngrid - Int32(1)
                w[i] = 1.0f0
            else
                t = (x - x_min) / x_step
                idx = Int32(floor(t)) + Int32(1)
                i_lo[i] = idx
                w[i] = t - Float32(idx - Int32(1))
            end
            i += stride
        end
    end
end

struct GPUGridInterpolator{N,T}
    i_lo::NTuple{N,CuArray{Int32,1}}
    w::NTuple{N,CuArray{T,1}}
    ngrid::NTuple{N,Int32}
    qsize::NTuple{N,Int32}
    qstride::NTuple{N,Int32}
    in_stride::NTuple{N,Int32}
end

function _strides(sizes::NTuple{N,Int32}) where {N}
    strides = Vector{Int32}(undef, N)
    s = Int32(1)
    for i in 1:N
        strides[i] = s
        s *= sizes[i]
    end
    return Tuple(strides)
end

"""
    gpu_linear_interpolator(grids, q_grids; T=Float32, threads=256)

    Build a reusable ND linear interpolator on GPU for uniform grids. The interpolator
    precomputes indices/weights for query grids `q_grids` and can be applied to
    different value arrays `U` defined on `grids`.
"""
function gpu_linear_interpolator(grids::NTuple{N,AbstractVector},
    q_grids::NTuple{N,AbstractVector}; T=Float32, threads::Int=256) where {N}
    ngrid = ntuple(d -> Int32(length(grids[d])), N)
    qsize = ntuple(d -> Int32(length(q_grids[d])), N)
    in_stride = _strides(ngrid)
    qstride = _strides(qsize)

    i_lo = ntuple(d -> CUDA.zeros(Int32, Int(qsize[d])), N)
    w = ntuple(d -> CUDA.zeros(T, Int(qsize[d])), N)

    for d in 1:N
        g = grids[d]
        q = q_grids[d]
        g_host = g isa CuArray ? Array(g) : g
        n = length(g_host)
        n < 2 && error("grid dimension $d must have at least 2 points")
        x_min = T(first(g_host))
        x_max = T(last(g_host))
        x_step = (x_max - x_min) / T(n - 1)
        q_dev = CuArray(T.(q))
        blocks = ceil(Int, length(q) / threads)
        @cuda threads=threads blocks=blocks interp_weights_kernel!(
            i_lo[d], w[d], q_dev, x_min, x_step, Int32(n)
        )
    end

    return GPUGridInterpolator{N,T}(i_lo, w, ngrid, qsize, qstride, in_stride)
end

@generated function interp_nd_kernel!(out, U, i_lo, w, qsize, qstride, in_stride, ::Val{N}) where {N}
    quote
        total = Int32(length(out))
        i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
        stride = gridDim().x * blockDim().x
        T = eltype(out)
        @inbounds begin
            while i <= total
                idx = i - Int32(1)
                Base.Cartesian.@nexprs $N d -> begin
                    q_idx_{d} = Int32(div(idx, qstride[d])) % qsize[d] + Int32(1)
                    i_lo_{d} = i_lo[d][q_idx_{d}]
                    w_{d} = w[d][q_idx_{d}]
                end

                val = zero(T)
                for mask in 0:Int32((1 << $N) - 1)
                    weight = one(T)
                    lin = Int32(1)
                    Base.Cartesian.@nexprs $N d -> begin
                        bit = (mask >> (d - 1)) & Int32(1)
                        w_use = ifelse(bit == Int32(1), w_{d}, one(T) - w_{d})
                        weight *= w_use
                        idx_d = i_lo_{d} + bit
                        lin += (idx_d - Int32(1)) * in_stride[d]
                    end
                    val += weight * U[Int(lin)]
                end

                out[i] = val
                i += stride
            end
        end
        return
    end
end

function interpolate!(out, U, itp::GPUGridInterpolator{N,T}; threads::Int=256) where {N,T}
    ndims(U) == N || error("U must have $N dimensions")
    eltype(U) == T || error("U element type must be $T")
    for d in 1:N
        size(U, d) == itp.ngrid[d] || error("U size mismatch on dimension $d")
        size(out, d) == itp.qsize[d] || error("out size mismatch on dimension $d")
    end
    blocks = ceil(Int, length(out) / threads)
    @cuda threads=threads blocks=blocks interp_nd_kernel!(
        out, U, itp.i_lo, itp.w, itp.qsize, itp.qstride, itp.in_stride, Val(N)
    )
    return out
end

function interpolate(U, itp::GPUGridInterpolator{N,T}; threads::Int=256) where {N,T}
    out = CUDA.zeros(T, map(Int, itp.qsize)...)
    return interpolate!(out, U, itp; threads=threads)
end

(itp::GPUGridInterpolator)(U; threads::Int=256) = interpolate(U, itp; threads=threads)

end
