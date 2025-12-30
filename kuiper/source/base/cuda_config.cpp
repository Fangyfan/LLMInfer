#include "base/cuda_config.h"

namespace kernel {
CudaConfig::CudaConfig(cudaStream_t stream) : stream_(stream) {}

CudaConfig::~CudaConfig() {
    if (stream_) {
        cudaError_t err = cudaStreamDestroy(stream_);
        CHECK_EQ(err, cudaSuccess) << "Failed to destroy CUDA stream: " << cudaGetErrorString(err);
    }
}

void CudaConfig::create() {
    CHECK_EQ(stream_, nullptr);
    cudaError_t err = cudaStreamCreate(&stream_);
    CHECK_EQ(err, cudaSuccess) << "Failed to create CUDA stream: " << cudaGetErrorString(err);
}

cudaStream_t CudaConfig::stream() const {
    return stream_;
}

void CudaConfig::set_stream(cudaStream_t stream) {
    stream_ = stream;
}
}  // namespace kernel