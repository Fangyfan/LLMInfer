#include "base/cuda_config.h"

namespace kernel {
CudaConfig::~CudaConfig() {
    if (stream) {
        cudaError_t err = cudaStreamDestroy(stream);
        CHECK_EQ(err, cudaSuccess) << "Failed to destroy CUDA stream: " << cudaGetErrorString(err) << std::endl;
    }
}

void CudaConfig::create() {
    CHECK_EQ(stream, nullptr);
    cudaError_t err = cudaStreamCreate(&stream);
    CHECK_EQ(err, cudaSuccess) << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
}
}  // namespace kernel