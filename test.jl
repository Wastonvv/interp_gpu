using Test
using CUDA
using Interpolations

include("Interp_gpu.jl")

CUDA.allowscalar(false)

function assert_close(out, expected; atol=5f-6)
    @test size(out) == size(expected)
    diff = maximum(abs.(out .- expected))
    @test diff <= atol
end

@testset "InterpGPU" begin
    @test CUDA.functional()

    @testset "1D linear interpolation (exact on linear data)" begin
        x = collect(Float32, range(-1, 1; length=8))
        q = collect(Float32, range(-1.2, 1.2; length=25))
        u = 2f0 .* x .+ 3f0

        itp = InterpGPU.gpu_linear_interpolator((x,), (q,); T=Float32)
        u_dev = CuArray(u)
        out = Array(InterpGPU.interpolate(u_dev, itp))

        itp_cpu = Interpolations.interpolate((x,), u, Gridded(Linear()))
        ext = extrapolate(itp_cpu, Flat())
        expected = Float32[ext(xq) for xq in q]
        assert_close(out, expected)
    end

    @testset "2D linear interpolation (matches CPU)" begin
        x = collect(Float32, range(-1, 1; length=6))
        y = collect(Float32, range(0, 2; length=5))
        qx = collect(Float32, range(-1.2, 1.2; length=7))
        qy = collect(Float32, range(-0.2, 2.2; length=9))

        u = Array{Float32}(undef, length(x), length(y))
        for i in eachindex(x), j in eachindex(y)
            u[i, j] = 1.5f0 * x[i] - 0.5f0 * y[j] + 2f0
        end

        itp = InterpGPU.gpu_linear_interpolator((x, y), (qx, qy); T=Float32)
        u_dev = CuArray(u)
        out = Array(InterpGPU.interpolate(u_dev, itp))

        itp_cpu = Interpolations.interpolate((x, y), u, Gridded(Linear()))
        ext = extrapolate(itp_cpu, (Flat(), Flat()))
        expected = Array{Float32}(undef, length(qx), length(qy))
        for i in eachindex(qx), j in eachindex(qy)
            expected[i, j] = ext(qx[i], qy[j])
        end

        assert_close(out, expected)
    end

    @testset "3D linear interpolation (matches CPU)" begin
        x = collect(Float32, range(-1, 1; length=5))
        y = collect(Float32, range(0, 2; length=4))
        z = collect(Float32, range(-2, 1; length=6))
        qx = collect(Float32, range(-1.1, 1.1; length=6))
        qy = collect(Float32, range(-0.3, 2.3; length=5))
        qz = collect(Float32, range(-2.2, 1.2; length=7))

        u = Array{Float32}(undef, length(x), length(y), length(z))
        for i in eachindex(x), j in eachindex(y), k in eachindex(z)
            u[i, j, k] = 0.7f0 * x[i] - 1.2f0 * y[j] + 0.4f0 * z[k] + 1.0f0
        end

        itp = InterpGPU.gpu_linear_interpolator((x, y, z), (qx, qy, qz); T=Float32)
        u_dev = CuArray(u)
        out = Array(InterpGPU.interpolate(u_dev, itp))

        itp_cpu = Interpolations.interpolate((x, y, z), u, Gridded(Linear()))
        ext = extrapolate(itp_cpu, (Flat(), Flat(), Flat()))
        expected = Array{Float32}(undef, length(qx), length(qy), length(qz))
        for i in eachindex(qx), j in eachindex(qy), k in eachindex(qz)
            expected[i, j, k] = ext(qx[i], qy[j], qz[k])
        end

        assert_close(out, expected)
    end

    @testset "interpolate! writes to provided output" begin
        x = collect(Float32, range(0, 1; length=4))
        q = collect(Float32, range(0, 1; length=6))
        u = 3f0 .* x .- 1f0

        itp = InterpGPU.gpu_linear_interpolator((x,), (q,); T=Float32)
        u_dev = CuArray(u)
        out_dev = CUDA.zeros(Float32, length(q))

        InterpGPU.interpolate!(out_dev, u_dev, itp)
        out = Array(out_dev)

        itp_cpu = Interpolations.interpolate((x,), u, Gridded(Linear()))
        ext = extrapolate(itp_cpu, Flat())
        expected = Float32[ext(xq) for xq in q]
        assert_close(out, expected)
    end
end
