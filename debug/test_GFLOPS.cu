#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device;
    cudaDeviceProp prop;

    // Get the current device
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int numSMs = prop.multiProcessorCount;      // Number of SMs
    int clockRate = prop.clockRate;            // Clock rate in kHz
    int coresPerSM;

    // Calculate cores per SM based on the GPU architecture
    switch (prop.major) {
        case 7: // Volta or Turing
            coresPerSM = (prop.minor == 0) ? 64 : 64;
            break;
        case 8: // Ampere
            coresPerSM = (prop.minor == 0) ? 64 : 128;
            break;
        case 9: // Ada Lovelace
            coresPerSM = 128;
            break;
        default:
            std::cerr << "Unknown or unsupported architecture!\n";
            return -1;
    }

    // Total cores
    int totalCores = numSMs * coresPerSM;

    // Peak GFLOPS = (Total Cores) x (Clock Rate in GHz) x 2 (for FMA)
    double peakGFLOPS = (double)totalCores * (clockRate / 1e6) * 2;

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Total SMs: " << numSMs << "\n";
    std::cout << "Cores per SM: " << coresPerSM << "\n";
    std::cout << "Clock Rate (MHz): " << clockRate / 1e3 << "\n";
    std::cout << "Peak GFLOPS: " << peakGFLOPS << " GFLOPS\n";

    return 0;
}

