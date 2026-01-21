#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernel/kernel_interface.h"
#include "base/cuda_config.h"

TEST(test_op_matmul, matmul_fp32_cuda_stream) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();

    tensor::Tensor weight_cpu(base::DataType::DataTypeFp32, 4, 4, true, allocator_cpu);
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, 4, true, allocator_cpu);

    for (int32_t i = 0; i < weight_cpu.size(); i++) {
        weight_cpu.index<float>(i) = float(i);
    }
    for (int32_t i = 0; i < input_cpu.size(); i++) {
        input_cpu.index<float>(i) = float(i);
    }

    tensor::Tensor weight_cu = weight_cpu.clone();
    tensor::Tensor input_cu = input_cpu.clone();
    weight_cu.to_cuda();
    input_cu.to_cuda();
        
    tensor::Tensor output_cpu(base::DataType::DataTypeFp32, 4, true, allocator_cpu);
    tensor::Tensor output_cu(base::DataType::DataTypeFp32, 4, true, allocator_cu);
    auto cuda_config = std::make_unique<kernel::CudaConfig>();
    cuda_config->create();

    kernel::get_matmul_kernel(base::DeviceType::DeviceCPU)(input_cpu, weight_cpu, output_cpu, nullptr);
    kernel::get_matmul_kernel(base::DeviceType::DeviceCUDA)(input_cu, weight_cu, output_cu, cuda_config->stream);

    output_cu.to_cpu();
    for (int32_t i = 0; i < output_cu.size(); i++) {
        ASSERT_EQ(output_cu.index<float>(i), output_cpu.index<float>(i));
    }
}

TEST(test_op_matmul, matmul_fp32_cpu) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cpu(base::DataType::DataTypeFp32, 3, true, allocator_cpu);
    tensor::Tensor weight_cpu(base::DataType::DataTypeFp32, 3, 3, true, allocator_cpu);
    tensor::Tensor output_cpu(base::DataType::DataTypeFp32, 3, true, allocator_cpu);
    
    input_cpu.index<float>(0) = 1.f;
    input_cpu.index<float>(1) = 1.f;
    input_cpu.index<float>(2) = -1.f;
    
    for (int32_t i = 0; i < weight_cpu.size(); i++) {
        weight_cpu.index<float>(i) = float(i + 1);
    }
    
    kernel::get_matmul_kernel(base::DeviceType::DeviceCPU)(input_cpu, weight_cpu, output_cpu, nullptr);

    // [1, 2, 3]   [ 1 ]   [ 0 ]
    // [4, 5, 6] × [ 1 ] = [ 3 ]
    // [7, 8, 9]   [-1 ]   [ 6 ]

    ASSERT_EQ(output_cpu.index<float>(0), 0);
    ASSERT_EQ(output_cpu.index<float>(1), 3);
    ASSERT_EQ(output_cpu.index<float>(2), 6);
}

TEST(test_op_matmul, matmul_fp32_cuda_no_stream) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor input_cu(base::DataType::DataTypeFp32, 3, true, allocator_cpu);
    tensor::Tensor weight_cu(base::DataType::DataTypeFp32, 3, 3, true, allocator_cpu);
    
    input_cu.index<float>(0) = 1.f;
    input_cu.index<float>(1) = 1.f;
    input_cu.index<float>(2) = -1.f;
    
    for (int32_t i = 0; i < weight_cu.size(); i++) {
        weight_cu.index<float>(i) = float(i + 1);
    }

    input_cu.to_cuda();
    weight_cu.to_cuda();
    
    tensor::Tensor output_cu(base::DataType::DataTypeFp32, 3, true, allocator_cu);
    kernel::get_matmul_kernel(base::DeviceType::DeviceCUDA)(input_cu, weight_cu, output_cu, nullptr);
    
    // [1, 2, 3]   [ 1 ]   [ 0 ]
    // [4, 5, 6] × [ 1 ] = [ 3 ]
    // [7, 8, 9]   [-1 ]   [ 6 ]
    
    output_cu.to_cpu();
    ASSERT_EQ(output_cu.index<float>(0), 0);
    ASSERT_EQ(output_cu.index<float>(1), 3);
    ASSERT_EQ(output_cu.index<float>(2), 6);
}