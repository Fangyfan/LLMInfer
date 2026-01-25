#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>
#include "../source/op/kernel/kernel_interface.h"
#include "base/cuda_config.h"

TEST(test_op_rope, rope_cuda_no_stream1) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t dim = 256;
    int32_t head_dim = 64;
    int32_t kv_dim = 128;
    int32_t max_seq_len = 1024;
    int32_t pos = 123;

    tensor::Tensor q_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);
    tensor::Tensor k_cpu(base::DataType::DataTypeFp32, kv_dim, true, allocator_cpu);
    tensor::Tensor token_pos(base::DataType::DataTypeInt32, 1, true, allocator_cpu);
    tensor::Tensor sin_cpu(base::DataType::DataTypeInt32, max_seq_len, head_dim / 2, true, allocator_cpu);
    tensor::Tensor cos_cpu(base::DataType::DataTypeInt32, max_seq_len, head_dim / 2, true, allocator_cpu);

    token_pos.index<int32_t>(0) = pos;

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (int32_t i = 0; i < dim; ++i) {
        q_cpu.index<float>(i) = dist(mt);
    }
    for (int32_t i = 0; i < kv_dim; ++i) {
        k_cpu.index<float>(i) = dist(mt);
    }

    kernel::get_sin_cos_cache_kernel(base::DeviceType::DeviceCPU)(sin_cpu, cos_cpu, head_dim, max_seq_len, nullptr);

    tensor::Tensor q_cu = q_cpu.clone();
    tensor::Tensor k_cu = k_cpu.clone();
    tensor::Tensor sin_cu = sin_cpu.clone();
    tensor::Tensor cos_cu = cos_cpu.clone();

    q_cu.to_cuda();
    k_cu.to_cuda();
    sin_cu.to_cuda();
    cos_cu.to_cuda();

    kernel::get_rope_kernel(base::DeviceType::DeviceCPU)(q_cpu, k_cpu, token_pos, sin_cpu, cos_cpu, dim, kv_dim, head_dim, nullptr);
    kernel::get_rope_kernel(base::DeviceType::DeviceCUDA)(q_cu, k_cu, token_pos, sin_cu, cos_cu, dim, kv_dim, head_dim, nullptr);
    q_cu.to_cpu();
    k_cu.to_cpu();

    for (int32_t i = 0; i < dim; ++i) {
        ASSERT_NEAR(q_cpu.index<float>(i), q_cu.index<float>(i), 1e-5f);
    }
    for (int32_t i = 0; i < kv_dim; ++i) {
        ASSERT_NEAR(k_cpu.index<float>(i), k_cu.index<float>(i), 1e-5f);
    }
}

TEST(test_op_rope, rope_cuda_no_stream2) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t dim = 1024;
    int32_t head_dim = 64;
    int32_t kv_dim = 256;
    int32_t max_seq_len = 2048;
    int32_t pos = 1234;

    tensor::Tensor q_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);
    tensor::Tensor k_cpu(base::DataType::DataTypeFp32, kv_dim, true, allocator_cpu);
    tensor::Tensor token_pos(base::DataType::DataTypeInt32, 1, true, allocator_cpu);
    tensor::Tensor sin_cpu(base::DataType::DataTypeInt32, max_seq_len, head_dim / 2, true, allocator_cpu);
    tensor::Tensor cos_cpu(base::DataType::DataTypeInt32, max_seq_len, head_dim / 2, true, allocator_cpu);

    token_pos.index<int32_t>(0) = pos;

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (int32_t i = 0; i < dim; ++i) {
        q_cpu.index<float>(i) = dist(mt);
    }
    for (int32_t i = 0; i < kv_dim; ++i) {
        k_cpu.index<float>(i) = dist(mt);
    }

    kernel::get_sin_cos_cache_kernel(base::DeviceType::DeviceCPU)(sin_cpu, cos_cpu, head_dim, max_seq_len, nullptr);

    tensor::Tensor q_cu = q_cpu.clone();
    tensor::Tensor k_cu = k_cpu.clone();
    tensor::Tensor sin_cu = sin_cpu.clone();
    tensor::Tensor cos_cu = cos_cpu.clone();

    q_cu.to_cuda();
    k_cu.to_cuda();
    sin_cu.to_cuda();
    cos_cu.to_cuda();

    kernel::get_rope_kernel(base::DeviceType::DeviceCPU)(q_cpu, k_cpu, token_pos, sin_cpu, cos_cpu, dim, kv_dim, head_dim, nullptr);
    kernel::get_rope_kernel(base::DeviceType::DeviceCUDA)(q_cu, k_cu, token_pos, sin_cu, cos_cu, dim, kv_dim, head_dim, nullptr);
    q_cu.to_cpu();
    k_cu.to_cpu();

    for (int32_t i = 0; i < dim; ++i) {
        ASSERT_NEAR(q_cpu.index<float>(i), q_cu.index<float>(i), 1e-5f);
    }
    for (int32_t i = 0; i < kv_dim; ++i) {
        ASSERT_NEAR(k_cpu.index<float>(i), k_cu.index<float>(i), 1e-5f);
    }
}

TEST(test_op_rope, rope_cuda_stream) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t dim = 512;
    int32_t head_dim = 64;
    int32_t kv_dim = 128;
    int32_t max_seq_len = 2048;
    int32_t pos = 1234;

    tensor::Tensor q_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);
    tensor::Tensor k_cpu(base::DataType::DataTypeFp32, kv_dim, true, allocator_cpu);
    tensor::Tensor token_pos(base::DataType::DataTypeInt32, 1, true, allocator_cpu);
    tensor::Tensor sin_cpu(base::DataType::DataTypeInt32, max_seq_len, head_dim / 2, true, allocator_cpu);
    tensor::Tensor cos_cpu(base::DataType::DataTypeInt32, max_seq_len, head_dim / 2, true, allocator_cpu);

    token_pos.index<int32_t>(0) = pos;

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (int32_t i = 0; i < dim; ++i) {
        q_cpu.index<float>(i) = dist(mt);
    }
    for (int32_t i = 0; i < kv_dim; ++i) {
        k_cpu.index<float>(i) = dist(mt);
    }

    kernel::get_sin_cos_cache_kernel(base::DeviceType::DeviceCPU)(sin_cpu, cos_cpu, head_dim, max_seq_len, nullptr);

    tensor::Tensor q_cu = q_cpu.clone();
    tensor::Tensor k_cu = k_cpu.clone();
    tensor::Tensor sin_cu = sin_cpu.clone();
    tensor::Tensor cos_cu = cos_cpu.clone();

    q_cu.to_cuda();
    k_cu.to_cuda();
    sin_cu.to_cuda();
    cos_cu.to_cuda();

    auto cuda_config = std::make_shared<kernel::CudaConfig>();
    cuda_config->create();

    kernel::get_rope_kernel(base::DeviceType::DeviceCPU)(q_cpu, k_cpu, token_pos, sin_cpu, cos_cpu, dim, kv_dim, head_dim, nullptr);
    kernel::get_rope_kernel(base::DeviceType::DeviceCUDA)(q_cu, k_cu, token_pos, sin_cu, cos_cu, dim, kv_dim, head_dim, cuda_config->stream);
    q_cu.to_cpu();
    k_cu.to_cpu();

    for (int32_t i = 0; i < dim; ++i) {
        ASSERT_NEAR(q_cpu.index<float>(i), q_cu.index<float>(i), 1e-5f);
    }
    for (int32_t i = 0; i < kv_dim; ++i) {
        ASSERT_NEAR(k_cpu.index<float>(i), k_cu.index<float>(i), 1e-5f);
    }
}