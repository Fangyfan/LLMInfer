#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernel/kernel_interface.h"
#include "base/cuda_config.h"

TEST(test_op_embedding, embedding_cuda_no_stream1) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t vocab_size = 4;
    int32_t dim = 512;

    tensor::Tensor input(base::DataType::DataTypeInt32, 1, true, allocator_cpu);
    tensor::Tensor weight(base::DataType::DataTypeFp32, vocab_size, dim, true, allocator_cpu);
    tensor::Tensor output(base::DataType::DataTypeFp32, 1, dim, true, allocator_cpu);

    input.index<int32_t>(0) = 1;
    for (int32_t i = 0; i < vocab_size * dim; ++i) {
        weight.index<float>(i) = float(i);
    }
    weight.to_cuda();
    output.to_cuda();

    kernel::get_embedding_kernel(base::DeviceType::DeviceCUDA)(input, weight, output, nullptr);
    output.to_cpu();

    for (int32_t i = 0; i < dim; ++i) {
        ASSERT_EQ(output.index<float>(i), dim + i);
    }
}

TEST(test_op_embedding, embedding_cuda_no_stream2) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t vocab_size = 4;
    int32_t dim = 512;

    tensor::Tensor input_cpu(base::DataType::DataTypeInt32, 2, true, allocator_cpu);
    tensor::Tensor weight_cpu(base::DataType::DataTypeFp32, vocab_size, dim, true, allocator_cpu);
    tensor::Tensor output_cpu(base::DataType::DataTypeFp32, 2, dim, true, allocator_cpu);

    input_cpu.index<int32_t>(0) = 1;
    input_cpu.index<int32_t>(1) = 3;
    for (int32_t i = 0; i < vocab_size * dim; ++i) {
        weight_cpu.index<float>(i) = float(i);
    }
    tensor::Tensor weight_cu = weight_cpu.clone();
    tensor::Tensor output_cu = output_cpu.clone();
    weight_cu.to_cuda();
    output_cu.to_cuda();

    kernel::get_embedding_kernel(base::DeviceType::DeviceCPU)(input_cpu, weight_cpu, output_cpu, nullptr);
    kernel::get_embedding_kernel(base::DeviceType::DeviceCUDA)(input_cpu, weight_cu, output_cu, nullptr);
    output_cu.to_cpu();

    for (int32_t i = 0; i < dim; ++i) {
        ASSERT_EQ(output_cpu.index<float>(i), dim + i);
        ASSERT_EQ(output_cpu.index<float>(i), output_cu.index<float>(i));
        ASSERT_EQ(output_cpu.index<float>(dim + i), 3 * dim + i);
        ASSERT_EQ(output_cpu.index<float>(dim + i), output_cu.index<float>(dim + i));
    }
}

TEST(test_op_embedding, embedding_cuda_stream) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t vocab_size = 4;
    int32_t dim = 512;

    tensor::Tensor input_cpu(base::DataType::DataTypeInt32, 2, true, allocator_cpu);
    tensor::Tensor weight_cpu(base::DataType::DataTypeFp32, vocab_size, dim, true, allocator_cpu);
    tensor::Tensor output_cpu(base::DataType::DataTypeFp32, 2, dim, true, allocator_cpu);

    input_cpu.index<int32_t>(0) = 1;
    input_cpu.index<int32_t>(1) = 3;
    for (int32_t i = 0; i < vocab_size * dim; ++i) {
        weight_cpu.index<float>(i) = float(i);
    }
    tensor::Tensor weight_cu = weight_cpu.clone();
    tensor::Tensor output_cu = output_cpu.clone();
    weight_cu.to_cuda();
    output_cu.to_cuda();

    auto cuda_config = std::make_shared<kernel::CudaConfig>();
    cuda_config->create();

    kernel::get_embedding_kernel(base::DeviceType::DeviceCPU)(input_cpu, weight_cpu, output_cpu, nullptr);
    kernel::get_embedding_kernel(base::DeviceType::DeviceCUDA)(input_cpu, weight_cu, output_cu, nullptr);
    output_cu.to_cpu();

    for (int32_t i = 0; i < dim; ++i) {
        ASSERT_EQ(output_cpu.index<float>(i), dim + i);
        ASSERT_EQ(output_cpu.index<float>(i), output_cu.index<float>(i));
        ASSERT_EQ(output_cpu.index<float>(dim + i), 3 * dim + i);
        ASSERT_EQ(output_cpu.index<float>(dim + i), output_cu.index<float>(dim + i));
    }
}