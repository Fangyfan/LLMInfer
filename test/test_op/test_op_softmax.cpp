#include <random>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernel/kernel_interface.h"
#include "base/cuda_config.h"

TEST(test_op_softmax, softmax_cuda_no_stream) {
    int32_t size = 32 * 15;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
        input_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor input_cu = input_cpu.clone();
    input_cu.to_cuda();
    
    kernel::get_softmax_kernel(base::DeviceType::DeviceCPU)(input_cpu, nullptr);
    kernel::get_softmax_kernel(base::DeviceType::DeviceCUDA)(input_cu, nullptr);
    
    input_cu.to_cpu();
    for (int32_t i = 0; i < size; ++i) {
        ASSERT_NEAR(input_cpu.index<float>(i), input_cu.index<float>(i), 1e-5f);
    }
}

TEST(test_op_softmax, softmax_cuda_stream) {
    int32_t size = 32 * 15;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
        input_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor input_cu = input_cpu.clone();
    input_cu.to_cuda();

    auto cuda_config = std::make_shared<kernel::CudaConfig>();
    cuda_config->create();
    
    kernel::get_softmax_kernel(base::DeviceType::DeviceCPU)(input_cpu, nullptr);
    kernel::get_softmax_kernel(base::DeviceType::DeviceCUDA)(input_cu, cuda_config->stream);
    
    input_cu.to_cpu();
    for (int32_t i = 0; i < size; ++i) {
        ASSERT_NEAR(input_cpu.index<float>(i), input_cu.index<float>(i), 1e-5f);
    }
}

TEST(test_op_softmax, softmax_cuda_stream2) {
    int32_t size = 32 * 151 * 15;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
        input_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor input_cu = input_cpu.clone();
    input_cu.to_cuda();

    auto cuda_config = std::make_shared<kernel::CudaConfig>();
    cuda_config->create();
    
    kernel::get_softmax_kernel(base::DeviceType::DeviceCPU)(input_cpu, nullptr);
    kernel::get_softmax_kernel(base::DeviceType::DeviceCUDA)(input_cu, cuda_config->stream);
    
    input_cu.to_cpu();
    for (int32_t i = 0; i < size; ++i) {
        ASSERT_NEAR(input_cpu.index<float>(i), input_cu.index<float>(i), 1e-5f);
    }
}