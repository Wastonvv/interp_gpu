# InterpGPU.jl

A high-performance GPU-accelerated N-dimensional linear interpolation library for Julia, designed for uniform grids.

## Overview

InterpGPU provides efficient linear interpolation on GPUs using CUDA. It's optimized for cases where you need to:
- Interpolate data on uniform grids in 1D, 2D, 3D, or higher dimensions
- Evaluate the same interpolant at many different query points
- Leverage GPU acceleration for batch interpolation operations

## Features

- **N-dimensional support**: Handle 1D, 2D, 3D, and higher-dimensional grids
- **GPU acceleration**: Uses CUDA.jl for fast parallel computation
- **Reusable interpolators**: Precompute indices and weights once, apply to multiple value arrays
- **Linear interpolation**: Fast O(1) lookup with linear basis functions
- **Boundary handling**: Flat extrapolation for points outside the grid domain
- **Flexible precision**: Support for both Float32 and Float64 computations

## Installation

Make sure you have CUDA.jl installed and a compatible GPU:

```julia
using Pkg
Pkg.add("CUDA")
include("Interp_gpu.jl")
```

## Usage

### Basic 1D Example

```julia
using CUDA
include("Interp_gpu.jl")

# Define grid and query points
x = collect(Float32, range(-1, 1; length=8))
q = collect(Float32, range(-1.2, 1.2; length=25))

# Define values on the grid
u = 2f0 .* x .+ 3f0

# Create the GPU interpolator
itp = InterpGPU.gpu_linear_interpolator((x,), (q,); T=Float32)

# Interpolate
u_dev = CuArray(u)
result = Array(InterpGPU.interpolate(u_dev, itp))
```

### 2D Example

```julia
# Define 2D grids
x = collect(Float32, range(-1, 1; length=6))
y = collect(Float32, range(0, 2; length=5))
qx = collect(Float32, range(-1.2, 1.2; length=7))
qy = collect(Float32, range(-0.2, 2.2; length=9))

# Define values on 2D grid
u = [1.5f0 * x[i] - 0.5f0 * y[j] + 2f0 for i in eachindex(x), j in eachindex(y)]

# Create and apply interpolator
itp = InterpGPU.gpu_linear_interpolator((x, y), (qx, qy); T=Float32)
u_dev = CuArray(u)
result = Array(InterpGPU.interpolate(u_dev, itp))
```

## API Reference

### `gpu_linear_interpolator(grids, q_grids; T=Float32, threads=256)`

Builds a reusable N-dimensional linear interpolator on GPU for uniform grids.

**Parameters:**
- `grids::NTuple{N,AbstractVector}`: Tuple of grid vectors (one per dimension)
- `q_grids::NTuple{N,AbstractVector}`: Tuple of query grid vectors
- `T::DataType`: Element type (Float32 or Float64), default: Float32
- `threads::Int`: CUDA thread count per block, default: 256

**Returns:** A `GPUGridInterpolator` object

### `interpolate(U, itp::GPUGridInterpolator{N,T})`

Evaluates the interpolant at all query points.

**Parameters:**
- `U::CuArray`: Values on the input grid
- `itp::GPUGridInterpolator`: Precomputed interpolator

**Returns:** Interpolated values as a CuArray

### `interpolate!(out, U, itp::GPUGridInterpolator{N,T})`

In-place version that writes results to a pre-allocated output array.

**Parameters:**
- `out::CuArray`: Output array (pre-allocated)
- `U::CuArray`: Values on the input grid
- `itp::GPUGridInterpolator`: Precomputed interpolator

## Performance Characteristics

- **Preprocessing**: O(Q·D) where Q is total query points and D is dimensions
- **Interpolation**: O(Q·2^D) FLOPS for linear interpolation
- **Memory**: Stores precomputed indices and weights on GPU for fast lookup

## Testing

Run the test suite with:

```julia
include("test.jl")
```

Tests verify correctness against CPU implementations using the Interpolations.jl library for 1D, 2D, and 3D cases.

## Requirements

- Julia 1.6+
- CUDA.jl (with a compatible GPU)
- CUDA-capable GPU (tested with NVIDIA GPUs)

## License

TBD

