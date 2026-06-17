#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int d = 0; d < D; d++) {
            sum += Q[row * D + d] * K[col * D + d];
        }

        scores[row * N + col] = sum / sqrtf((float)D);
    }
}

__global__ void softmax_rows_kernel(float* scores, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= N) return;

    // Compute row maximum
    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[row * N + j]);
    }

    __shared__ float smax[256];
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = smax[0];

    // Compute exponentials and partial sums
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float val = expf(scores[row * N + j] - row_max);
        scores[row * N + j] = val;
        local_sum += val;
    }

    __shared__ float ssum[256];
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = ssum[0];

    // Normalize
    for (int j = tid; j < N; j += blockDim.x) {
        scores[row * N + j] /= row_sum;
    }
}

__global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < D) {
        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum += attn[row * N + k] * V[k * D + col];
        }

        output[row * D + col] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int D) {
    float* scores;
    cudaMalloc(&scores, (size_t)N * N * sizeof(float));

    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);
    scores_kernel<<<sBlocks, sThreads>>>(Q, K, scores, N, D);

    softmax_rows_kernel<<<N, 256>>>(scores, N);

    dim3 oThreads(16, 16);
    dim3 oBlocks((D + 15) / 16, (N + 15) / 16);
    av_kernel<<<oBlocks, oThreads>>>(scores, V, output, N, D);

    cudaDeviceSynchronize();
    cudaFree(scores);
}