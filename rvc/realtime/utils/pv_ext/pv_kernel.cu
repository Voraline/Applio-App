#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

template <typename T>
__device__ __forceinline__ T device_cos(T x) {
    return cos(x);
}

template <>
__device__ __forceinline__ float device_cos<float>(float x) {
    return __cosf(x);
}

template <typename scalar_t>
__global__ void pv_synthesize_kernel(
    scalar_t* __restrict__       out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ fade_out_sq,
    const scalar_t* __restrict__ fade_in_sq,
    const scalar_t* __restrict__ window_over_n,
    const scalar_t* __restrict__ t_val,
    const scalar_t* __restrict__ kg_d,
    const scalar_t* __restrict__ phia,
    const scalar_t* __restrict__ absab,
    const int T,
    const int K
) {
    extern __shared__ char smem_raw[];
    scalar_t* s_kgd   = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* s_phia  = s_kgd   + K;
    scalar_t* s_absab = s_phia  + K;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        s_kgd[k]   = __ldg(&kg_d[k]);
        s_phia[k]  = __ldg(&phia[k]);
        s_absab[k] = __ldg(&absab[k]);
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    scalar_t tv  = __ldg(&t_val[t]);
    scalar_t acc = static_cast<scalar_t>(0);

    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        acc += device_cos(tv * s_kgd[k] + s_phia[k]) * s_absab[k];
    }

    out[t] = __ldg(&a[t]) * __ldg(&fade_out_sq[t])
           + __ldg(&b[t]) * __ldg(&fade_in_sq[t])
           + acc * __ldg(&window_over_n[t]);
}

template <typename scalar_t>
__global__ void pv_synthesize_global_kernel(
    scalar_t* __restrict__       out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ fade_out_sq,
    const scalar_t* __restrict__ fade_in_sq,
    const scalar_t* __restrict__ window_over_n,
    const scalar_t* __restrict__ t_val,
    const scalar_t* __restrict__ kg_d,
    const scalar_t* __restrict__ phia,
    const scalar_t* __restrict__ absab,
    const int T,
    const int K
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    scalar_t tv  = __ldg(&t_val[t]);
    scalar_t acc = static_cast<scalar_t>(0);

    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        acc += device_cos(tv * __ldg(&kg_d[k]) + __ldg(&phia[k])) * __ldg(&absab[k]);
    }

    out[t] = __ldg(&a[t]) * __ldg(&fade_out_sq[t])
           + __ldg(&b[t]) * __ldg(&fade_in_sq[t])
           + acc * __ldg(&window_over_n[t]);
}

torch::Tensor pv_synthesize_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor fade_out_sq,
    torch::Tensor fade_in_sq,
    torch::Tensor window_over_n,
    torch::Tensor t_over_n,
    torch::Tensor k_grid_delta,
    torch::Tensor phia,
    torch::Tensor absab
) {
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto fo_c = fade_out_sq.contiguous();
    auto fi_c = fade_in_sq.contiguous();
    auto wn_c = window_over_n.contiguous();
    auto ton_c = t_over_n.contiguous();
    auto kgd_c = k_grid_delta.contiguous();
    auto phia_c = phia.contiguous();
    auto absab_c = absab.contiguous();

    const int T = static_cast<int>(ton_c.size(0));
    const int K = static_cast<int>(kgd_c.size(0));

    auto out = torch::empty({T}, a_c.options());

    constexpr int BLOCK_SIZE = 256;
    const int grid = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const size_t smem_bytes = static_cast<size_t>(3 * K) * sizeof(float);

    int device_id = a_c.device().index();
    if (device_id < 0) {
        cudaGetDevice(&device_id);
    }

    int max_smem = 48 * 1024;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device_id);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id).stream();

    AT_DISPATCH_FLOATING_TYPES(a_c.scalar_type(), "pv_synthesize_cuda", ([&] {
        if (smem_bytes <= static_cast<size_t>(max_smem)) {
            pv_synthesize_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
                out.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                b_c.data_ptr<scalar_t>(),
                fo_c.data_ptr<scalar_t>(),
                fi_c.data_ptr<scalar_t>(),
                wn_c.data_ptr<scalar_t>(),
                ton_c.data_ptr<scalar_t>(),
                kgd_c.data_ptr<scalar_t>(),
                phia_c.data_ptr<scalar_t>(),
                absab_c.data_ptr<scalar_t>(),
                T, K
            );
        } else {
            pv_synthesize_global_kernel<scalar_t><<<grid, BLOCK_SIZE, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                b_c.data_ptr<scalar_t>(),
                fo_c.data_ptr<scalar_t>(),
                fi_c.data_ptr<scalar_t>(),
                wn_c.data_ptr<scalar_t>(),
                ton_c.data_ptr<scalar_t>(),
                kgd_c.data_ptr<scalar_t>(),
                phia_c.data_ptr<scalar_t>(),
                absab_c.data_ptr<scalar_t>(),
                T, K
            );
        }
    }));

    return out;
}
