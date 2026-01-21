#include <random>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernel/kernel_interface.h"
#include "../utils.cuh"

TEST(test_op_rmsnorm, rmsnorm_cuda_no_stream) {
    int32_t size = 32 * 15;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor weight_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor output_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int32_t i = 0; i < size; i++) {
        input_cpu.index<float>(i) = dist(mt);
        weight_cpu.index<float>(i) = dist(mt);
    }
    // LOG(INFO) << "input_cpu[" << 0 << "] = " << input_cpu.index<float>(0) << std::endl;
    // LOG(INFO) << "weight_cpu[" << 0 << "] = " << weight_cpu.index<float>(0) << std::endl;

    tensor::Tensor input_cu = input_cpu.clone();
    tensor::Tensor weight_cu = weight_cpu.clone();
    tensor::Tensor output_cu = output_cpu.clone();

    input_cu.to_cuda();
    weight_cu.to_cuda();
    output_cu.to_cuda();
    
    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCUDA)(input_cu, weight_cu, output_cu, nullptr);
    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCPU)(input_cpu, weight_cpu, output_cpu, nullptr);
    
    output_cu.to_cpu();
    for (int32_t i = 0; i < size; i++) {
        ASSERT_NEAR(output_cpu.index<float>(i), output_cu.index<float>(i), 1e-5f);
    }
}

TEST(test_op_rmsnorm, rmsnorm_cuda_stream) {
    int32_t size = 32 * 15;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor weight_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor output_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int32_t i = 0; i < size; i++) {
        input_cpu.index<float>(i) = dist(mt);
        weight_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor input_cu = input_cpu.clone();
    tensor::Tensor weight_cu = weight_cpu.clone();
    tensor::Tensor output_cu = output_cpu.clone();

    input_cu.to_cuda();
    weight_cu.to_cuda();
    output_cu.to_cuda();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCUDA)(input_cu, weight_cu, output_cu, stream);
    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCPU)(input_cpu, weight_cpu, output_cpu, nullptr);
    
    output_cu.to_cpu();
    for (int32_t i = 0; i < size; i++) {
        ASSERT_NEAR(output_cpu.index<float>(i), output_cu.index<float>(i), 1e-5f);
    }
    cudaStreamDestroy(stream);
}

TEST(test_op_rmsnorm, rmsnorm_cuda_stream2) {
    int32_t size = 32 * 151 * 15;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor weight_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor output_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int32_t i = 0; i < size; i++) {
        input_cpu.index<float>(i) = dist(mt);
        weight_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor input_cu = input_cpu.clone();
    tensor::Tensor weight_cu = weight_cpu.clone();
    tensor::Tensor output_cu = output_cpu.clone();

    input_cu.to_cuda();
    weight_cu.to_cuda();
    output_cu.to_cuda();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCUDA)(input_cu, weight_cu, output_cu, stream);
    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCPU)(input_cpu, weight_cpu, output_cpu, nullptr);
    
    output_cu.to_cpu();
    for (int32_t i = 0; i < size; i++) {
        ASSERT_NEAR(output_cpu.index<float>(i), output_cu.index<float>(i), 1e-5f);
    }
    cudaStreamDestroy(stream);
}