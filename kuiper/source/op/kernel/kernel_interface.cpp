#include "kernel_interface.h"
#include "cpu/add_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/swiglu_kernel.h"
#include "cpu/embedding_kernel.h"
#include "cuda/add_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/matmul_kernel.cuh"
#include "cuda/swiglu_kernel.cuh"
#include "cuda/embedding_kernel.cuh"

namespace kernel {
AddKernel get_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCPU) {
        return add_kernel_cpu;
    } else if (device_type == base::DeviceType::DeviceCUDA) {
        return add_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a add kernel." << std::endl;
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCPU) {
        return rmsnorm_kernel_cpu;
    } else if (device_type == base::DeviceType::DeviceCUDA) {
        return rmsnorm_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel." << std::endl;
        return nullptr;
    }
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCPU) {
        return matmul_kernel_cpu;
    } else if (device_type == base::DeviceType::DeviceCUDA) {
        return matmul_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel." << std::endl;
        return nullptr;
    }
}

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return matmul_kernel_cu_quant8;
    } else {
        LOG(FATAL) << "Unknown device type for get a quant8 matmul kernel." << std::endl;
        return nullptr;
    }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCPU) {
        return swiglu_kernel_cpu;
    } else if (device_type == base::DeviceType::DeviceCUDA) {
        return swiglu_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a swiglu kernel." << std::endl;
        return nullptr;
    }
}

EmbeddingKernel get_embedding_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCPU) {
        return embedding_kernel_cpu;
    } else if (device_type == base::DeviceType::DeviceCUDA) {
        return embedding_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a embedding kernel." << std::endl;
        return nullptr;
    }
}
}  // namespace kernel