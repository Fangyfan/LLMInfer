#include <random>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernel/kernel_interface.h"

TEST(test_op_rmsnorm, rmsnorm_cuda_no_stream) {
    int32_t size = 32 * 15;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor weight_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor output_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
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
    for (int32_t i = 0; i < size; ++i) {
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
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
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
    for (int32_t i = 0; i < size; ++i) {
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
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
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
    for (int32_t i = 0; i < size; ++i) {
        ASSERT_NEAR(output_cpu.index<float>(i), output_cu.index<float>(i), 1e-5f);
    }
    cudaStreamDestroy(stream);
}

TEST(test_op_rmsnorm, rmsnorm_2d_cuda_stream) {
    int32_t dims_size = 4;
    int32_t dim = 1024;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, dims_size, dim, true, allocator_cpu);
    tensor::Tensor weight_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);
    tensor::Tensor output_cpu(base::DataType::DataTypeFp32, dims_size, dim, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < dims_size; ++i) {
        for (int32_t j = 0; j < dim; ++j) {
            input_cpu.index<float>(i * dim + j) = dist(mt);
        }
    }
    for (int32_t i = 0; i < dim; ++i) {
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
    kernel::get_rmsnorm_2d_kernel(base::DeviceType::DeviceCUDA)(input_cu, weight_cu, output_cu, dim, stream);

    for (int32_t i = 0; i < dims_size; ++i) {
        tensor::Tensor input_cpu1(base::DataType::DataTypeFp32, dim, false, nullptr, input_cpu.ptr<float>(i * dim));
        input_cpu1.set_device_type(base::DeviceType::DeviceCPU);
        tensor::Tensor output_cpu1(base::DataType::DataTypeFp32, dim, false, nullptr, output_cpu.ptr<float>(i * dim));
        output_cpu1.set_device_type(base::DeviceType::DeviceCPU);
        kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCPU)(input_cpu1, weight_cpu, output_cpu1, nullptr);
    }
    
    output_cu.to_cpu();
    for (int32_t i = 0; i < dims_size; ++i) {
        for (int32_t j = 0; j < dim; ++j) {
            ASSERT_NEAR(output_cpu.index<float>(i * dim + j), output_cu.index<float>(i * dim + j), 1e-5f);
        }
    }
    cudaStreamDestroy(stream);
}