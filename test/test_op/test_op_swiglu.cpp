#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>
#include "../source/op/kernel/kernel_interface.h"
#include "base/cuda_config.h"

TEST(test_op_swiglu, swiglu_cuda_no_stream) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    
    int32_t size = 32 * 151;
    tensor::Tensor in1_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor in2_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor out_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
        in1_cpu.index<float>(i) = dist(mt);
        in2_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor in1_cu = in1_cpu.clone();
    tensor::Tensor in2_cu = in2_cpu.clone();
    tensor::Tensor out_cu = out_cpu.clone();
    in1_cu.to_cuda();
    in2_cu.to_cuda();
    out_cu.to_cuda();

    kernel::get_swiglu_kernel(base::DeviceType::DeviceCPU)(in1_cpu, in2_cpu, out_cpu, nullptr);
    kernel::get_swiglu_kernel(base::DeviceType::DeviceCUDA)(in1_cu, in2_cu, out_cu, nullptr);
    out_cu.to_cpu();

    for (int32_t i = 0; i < size; ++i) {
        ASSERT_NEAR(out_cpu.index<float>(i), out_cu.index<float>(i), 1e-5f);
    }
}

TEST(test_op_swiglu, swiglu_cuda_stream) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    
    int32_t size = 32 * 151;
    tensor::Tensor in1_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor in2_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor out_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
        in1_cpu.index<float>(i) = dist(mt);
        in2_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor in1_cu = in1_cpu.clone();
    tensor::Tensor in2_cu = in2_cpu.clone();
    tensor::Tensor out_cu = out_cpu.clone();
    in1_cu.to_cuda();
    in2_cu.to_cuda();
    out_cu.to_cuda();

    auto cuda_config = std::make_shared<kernel::CudaConfig>();
    cuda_config->create();

    kernel::get_swiglu_kernel(base::DeviceType::DeviceCPU)(in1_cpu, in2_cpu, out_cpu, nullptr);
    kernel::get_swiglu_kernel(base::DeviceType::DeviceCUDA)(in1_cu, in2_cu, out_cu, cuda_config->stream);
    out_cu.to_cpu();

    for (int32_t i = 0; i < size; ++i) {
        ASSERT_NEAR(out_cpu.index<float>(i), out_cu.index<float>(i), 1e-5f);
    }
}