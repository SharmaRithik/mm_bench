#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include "2d_tiling.cuh"

#define CHECK_CUDA(err) { cudaCheck((err), __FILE__, __LINE__); }
#define CHECK_LAST_CUDA_ERROR() { cudaCheck(cudaGetLastError(), __FILE__, __LINE__); }

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void randomize_matrix(float *mat, int N) {
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

bool verify_matrix(float *matRef, float *matOut, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (diff > 0.01) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                   matRef[i], matOut[i], diff, i);
            return false;
        }
    }
    return true;
}

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));

    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

int main(int argc, char **argv) {
    std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};
    int max_size = SIZE[SIZE.size() - 1];
    std::cout << "Max size: " << max_size << std::endl;
    
    float alpha = 0.5f;
    float beta = 3.0f;
    
    // Host memory allocation
    float *A = (float *)malloc(sizeof(float) * max_size * max_size);
    float *B = (float *)malloc(sizeof(float) * max_size * max_size);
    float *C = (float *)malloc(sizeof(float) * max_size * max_size);
    float *C_ref = (float *)malloc(sizeof(float) * max_size * max_size);
    if (!A || !B || !C || !C_ref) {
        printf("Host memory allocation failed\n");
        exit(1);
    }

    // Initialize matrices
    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);
    
    // Device memory allocation
    float *dA, *dB, *dC, *dC_ref;
    CHECK_CUDA(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
    CHECK_CUDA(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
    CHECK_CUDA(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
    CHECK_CUDA(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        exit(1);
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int repeat_times = 50;
    
    for (int size : SIZE) {
        int M = size;
        int N = size;
        int K = size;
        
        std::cout << "dimensions(m=n=k) " << M << ", alpha: " << alpha
                  << ", beta: " << beta << std::endl;
        
        // Copy current size data to device
        CHECK_CUDA(cudaMemcpy(dA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B, sizeof(float) * K * N, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC_ref, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));
        
        // Run reference cuBLAS
        cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, M, K,
                                   &alpha,
                                   dB, N,
                                   dA, K,
                                   &beta,
                                   dC_ref, N);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS GEMM failed\n");
            exit(1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
                    
        // Run our implementation once for correctness
        runSgemm2DBlocktiling(M, N, K, alpha, dA, dB, beta, dC);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy results back for verification
        CHECK_CUDA(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(C_ref, dC_ref, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        
        if (!verify_matrix(C_ref, C, M * N)) {
            printf("Verification failed for size %d!\n", size);
            exit(1);
        }
        
        // Timing runs
        CHECK_CUDA(cudaEventRecord(start));
        for (int j = 0; j < repeat_times; j++) {
            runSgemm2DBlocktiling(M, N, K, alpha, dA, dB, beta, dC);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        float seconds = milliseconds / 1000.0f;
        
        // Use double for higher precision in GFLOPS calculation
        double flops = (double)2.0 * M * N * K;
        double gflops = (repeat_times * flops * 1e-9) / seconds;
        
        printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: (%d).\n",
               seconds / repeat_times,
               gflops, M);
    }

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dC_ref));
    cublasDestroy(handle);
    free(A);
    free(B);
    free(C);
    free(C_ref);

    return 0;
}
