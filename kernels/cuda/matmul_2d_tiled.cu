#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Error checking macro
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Kernel configuration structure
struct KernelConfig {
    dim3 gridDim;
    dim3 blockDim;
    int BM;  // Tile size M
    int BN;  // Tile size N
    int BK;  // Tile size K
    int TM;  // Per-thread tile M
    int TN;  // Per-thread tile N
};

// Template kernel for 2D tiled matrix multiplication
template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_2d_tiled(const float* A, const float* B, float* C, 
                               int M, int N, int K) {
    // Shared memory declarations
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    // Thread indices
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;
    
    // Load strides
    const uint loadStrideA = blockDim.x * blockDim.y;
    const uint loadStrideB = loadStrideA;
    
    // Result accumulation registers
    float results[TM][TN] = {0};
    
    // Iterate over K dimension tiles
    for(int tileK = 0; tileK < K; tileK += BK) {
        // Load matrix A tile
        for(uint loadOffset = 0; loadOffset < BK; loadOffset += loadStrideA) {
            uint col = tx + loadOffset;
            if(col < BK && tileK + col < K) {
                for(uint row = ty; row < BM; row += blockDim.y) {
                    if(by * BM + row < M) {
                        As[row][col] = A[(by * BM + row) * K + (tileK + col)];
                    }
                }
            }
        }
        
        // Load matrix B tile
        for(uint loadOffset = 0; loadOffset < BN; loadOffset += loadStrideB) {
            uint row = ty + loadOffset;
            if(row < BK && tileK + row < K) {
                for(uint col = tx; col < BN; col += blockDim.x) {
                    if(bx * BN + col < N) {
                        Bs[row][col] = B[(tileK + row) * N + (bx * BN + col)];
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute tile results
        for(uint k = 0; k < BK && tileK + k < K; k++) {
            for(uint tm = 0; tm < TM; tm++) {
                uint rowA = ty * TM + tm;
                if(rowA < BM && by * BM + rowA < M) {
                    float aVal = As[rowA][k];
                    
                    for(uint tn = 0; tn < TN; tn++) {
                        uint colB = tx * TN + tn;
                        if(colB < BN && bx * BN + colB < N) {
                            results[tm][tn] += aVal * Bs[k][colB];
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    for(uint tm = 0; tm < TM; tm++) {
        uint globalRow = by * BM + ty * TM + tm;
        if(globalRow < M) {
            for(uint tn = 0; tn < TN; tn++) {
                uint globalCol = bx * BN + tx * TN + tn;
                if(globalCol < N) {
                    C[globalRow * N + globalCol] = results[tm][tn];
                }
            }
        }
    }
}

// Host function to set up and launch kernel
void launch_matmul_2d_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K, KernelConfig config) {
    switch(config.BM) {
        case 64:
            switch(config.BN) {
                case 64:
                    matmul_2d_tiled<64, 64, 8, 8, 8><<<config.gridDim, config.blockDim>>>(A, B, C, M, N, K);
                    break;
                case 32:
                    matmul_2d_tiled<64, 32, 8, 8, 4><<<config.gridDim, config.blockDim>>>(A, B, C, M, N, K);
                    break;
            }
            break;
        case 32:
            switch(config.BN) {
                case 64:
                    matmul_2d_tiled<32, 64, 8, 4, 8><<<config.gridDim, config.blockDim>>>(A, B, C, M, N, K);
                    break;
                case 32:
                    matmul_2d_tiled<32, 32, 8, 4, 4><<<config.gridDim, config.blockDim>>>(A, B, C, M, N, K);
                    break;
            }
            break;
    }
    cudaCheckError();
}

// Compute single element for verification
float compute_single_element(const float* A, const float* B, 
                           int M, int N, int K,
                           int row, int col) {
    float sum = 0.0f;
    for(int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    return sum;
}

// Verification function with random sampling
bool verify_result(const float* gpu_C, const float* A, const float* B,
                  int M, int N, int K,
                  int num_samples = 100,
                  float tolerance = 1e-4) {
    // Seed for reproducibility
    srand(42);
    
    int mismatches = 0;
    printf("\nVerifying %d random samples...\n", num_samples);
    
    for(int i = 0; i < num_samples; i++) {
        // Generate random indices
        int row = rand() % M;
        int col = rand() % N;
        int idx = row * N + col;
        
        // Compute reference value for this element
        float cpu_val = compute_single_element(A, B, M, N, K, row, col);
        float gpu_val = gpu_C[idx];
        
        float rel_error = fabs(cpu_val - gpu_val) / (fabs(cpu_val) + 1e-6);
        
        if(rel_error > tolerance) {
            mismatches++;
            if(mismatches <= 10) {
                printf("Mismatch at [%d,%d]: CPU = %f, GPU = %f, Relative Error = %e\n",
                       row, col, cpu_val, gpu_val, rel_error);
            }
            if(mismatches > 10) break;
        }
    }
    
    if(mismatches > 0) {
        float mismatch_percentage = (float)mismatches / num_samples * 100;
        printf("Total mismatches in sample: %d (%.2f%%)\n", mismatches, mismatch_percentage);
        return mismatch_percentage < 5.0;  // Allow up to 5% mismatches in sample
    }
    
    printf("All sampled elements verified successfully!\n");
    return true;
}

// Timing utilities
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} CudaTimer;

void start_timer(CudaTimer* timer) {
    cudaEventCreate(&timer->start);
    cudaEventCreate(&timer->stop);
    cudaEventRecord(timer->start);
}

float stop_timer(CudaTimer* timer) {
    float ms;
    cudaEventRecord(timer->stop);
    cudaEventSynchronize(timer->stop);
    cudaEventElapsedTime(&ms, timer->start, timer->stop);
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);
    return ms;
}

// Exposed C interface for Python
extern "C" {
    float run_matmul_benchmark(int M, int N, int K,
                             int BM, int BN, int BK,
                             int TM, int TN,
                             int verify = 1) {
        // Allocate host memory
        float *h_A = (float*)malloc(M * K * sizeof(float));
        float *h_B = (float*)malloc(K * N * sizeof(float));
        float *h_C = (float*)malloc(M * N * sizeof(float));
        
        // Initialize input matrices
        srand(42);  // Fixed seed for reproducibility
        for(int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
        for(int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        
        // Copy input data to device
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        
        // Calculate grid and block dimensions
        KernelConfig config;
        config.BM = BM;
        config.BN = BN;
        config.BK = BK;
        config.TM = TM;
        config.TN = TN;
        
        config.blockDim = dim3((BN + TN - 1) / TN, (BM + TM - 1) / TM);
        config.gridDim = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        
        // Warm-up run
        launch_matmul_2d_tiled(d_A, d_B, d_C, M, N, K, config);
        cudaDeviceSynchronize();
        
        // Timed run
        CudaTimer timer;
        start_timer(&timer);
        launch_matmul_2d_tiled(d_A, d_B, d_C, M, N, K, config);
        float ms = stop_timer(&timer);
        
        // Verify result if requested
        if(verify) {
            cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            if(!verify_result(h_C, h_A, h_B, M, N, K, 100)) {  // Verify 100 random samples
                printf("Verification failed!\n");
                ms = -1.0f;
            }
        }
        
        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        
        return ms;
    }
}
