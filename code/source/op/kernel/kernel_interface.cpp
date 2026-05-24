#include "kernel_interface.h"

#include "cpu/add_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/swiglu_kernel.h"
#include "cpu/embedding_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/scale_kernel.h"
#include "cpu/scale_sum_kernel.h"

#include "cuda/add_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/matmul_kernel.cuh"
#include "cuda/swiglu_kernel.cuh"
#include "cuda/embedding_kernel.cuh"
#include "cuda/rope_kernel.cuh"
#include "cuda/mha_kernel.cuh"
#include "cuda/argmax_kernel.cuh"

namespace kernel {
AddKernel get_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return add_kernel_cu;
    } else if (device_type == base::DeviceType::DeviceCPU) {
        return add_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a add kernel." << std::endl;
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return rmsnorm_kernel_cu;
    } else if (device_type == base::DeviceType::DeviceCPU) {
        return rmsnorm_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel." << std::endl;
        return nullptr;
    }
}

FusedAddRMSNormKernel get_fused_add_rmsnorm_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return fused_add_rmsnorm_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel." << std::endl;
        return nullptr;
    }
}

RMSNorm2DKernel get_rmsnorm_2d_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return rmsnorm_2d_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel." << std::endl;
        return nullptr;
    }
}

GEMVKernel get_gemv_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return gemv_kernel_cu;
    } else if (device_type == base::DeviceType::DeviceCPU) {
        return matmul_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel." << std::endl;
        return nullptr;
    }
}

FusedGEMVAddKernel get_fused_gemv_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return fused_gemv_add_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a fused_gemv_add kernel." << std::endl;
        return nullptr;
    }
}

FusedQKVGEMVKernel get_fused_qkv_gemv_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return fused_qkv_gemv_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a fused_qkv_gemv kernel." << std::endl;
        return nullptr;
    }
}

GEMVInt8Kernel get_gemv_int8_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return gemv_int8_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a gemv_int8 kernel." << std::endl;
        return nullptr;
    }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return swiglu_kernel_cu;
    } else if (device_type == base::DeviceType::DeviceCPU) {
        return swiglu_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a swiglu kernel." << std::endl;
        return nullptr;
    }
}

FusedGateUpSwiGLUKernel get_fused_gate_up_gemv_swiglu_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return fused_gate_up_gemv_swiglu_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a fused_gate_up_gemv_swiglu kernel." << std::endl;
        return nullptr;
    }
}

EmbeddingKernel get_embedding_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return embedding_kernel_cu;
    } else if (device_type == base::DeviceType::DeviceCPU) {
        return embedding_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a embedding kernel." << std::endl;
        return nullptr;
    }
}

SinCosCacheKernel get_sin_cos_cache_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return sin_cos_cache_precompute_cu;
    } else if (device_type == base::DeviceType::DeviceCPU) {
        return sin_cos_cache_precompute_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a sin/cos cache precompute kernel." << std::endl;
        return nullptr;
    }
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return rope_kernel_cu;
    } else if (device_type == base::DeviceType::DeviceCPU) {
        return rope_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rope kernel." << std::endl;
        return nullptr;
    }
}

FusedQKNormRoPEKernel get_fused_qk_norm_rope_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return fused_qk_norm_rope_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a fused_qk_norm_rope kernel." << std::endl;
        return nullptr;
    }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return mha_kernel_cu;
    } else if (device_type == base::DeviceType::DeviceCPU) {
        return mha_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a mha kernel." << std::endl;
        return nullptr;
    }
}

SoftmaxKernel get_softmax_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCUDA) {
        return softmax_kernel_cu;
    } else if (device_type == base::DeviceType::DeviceCPU) {
        return softmax_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a softmax kernel." << std::endl;
        return nullptr;
    }
}

ScaleKernel get_scale_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCPU) {
        return scale_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a scale kernel." << std::endl;
        return nullptr;
    }
}

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCPU) {
        return scale_sum_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a scale sum kernel." << std::endl;
        return nullptr;
    }
}
}  // namespace kernel