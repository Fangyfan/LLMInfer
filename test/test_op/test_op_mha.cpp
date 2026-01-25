#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>
#include "../source/op/kernel/kernel_interface.h"
#include "base/cuda_config.h"

TEST(test_op_mha, mha_cuda_no_stream1) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t dim = 256;
    int32_t kv_dim = 128;
    int32_t head_dim = 64;
    int32_t layer_num = 10;
    int32_t max_seq_len = 256;

    int32_t layer_id = 3;
    int32_t pos = 12;

    int32_t kv_mul = dim / kv_dim;
    int32_t head_num = dim / head_dim;

    tensor::Tensor q_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);
    tensor::Tensor s_cpu(base::DataType::DataTypeFp32, head_num, max_seq_len, true, allocator_cpu);
    tensor::Tensor k_cpu(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator_cpu);
    tensor::Tensor v_cpu(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator_cpu);
    tensor::Tensor o_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (int32_t i = 0; i < dim; ++i) {
        q_cpu.index<float>(i) = dist(mt);
    }
    for (int32_t l = 0; l < layer_num; ++l) {
        for (int32_t i = 0; i <= pos; ++i) {
            for (int32_t j = 0; j < kv_dim; ++j) {
                int32_t idx = l * max_seq_len * kv_dim + i * kv_dim + j;
                k_cpu.index<float>(idx) = dist(mt);
                v_cpu.index<float>(idx) = dist(mt);
            }
        }
    }
    allocator_cpu->memset_zero(o_cpu.ptr<float>(), o_cpu.byte_size(), nullptr);

    tensor::Tensor q_cu = q_cpu.clone();
    tensor::Tensor s_cu = s_cpu.clone();
    tensor::Tensor k_cu = k_cpu.clone();
    tensor::Tensor v_cu = v_cpu.clone();
    tensor::Tensor o_cu = o_cpu.clone();

    q_cu.to_cuda();
    s_cu.to_cuda();
    k_cu.to_cuda();
    v_cu.to_cuda();
    o_cu.to_cuda();

    kernel::get_mha_kernel(base::DeviceType::DeviceCPU)(q_cpu, s_cpu, k_cpu, v_cpu, o_cpu, layer_id, pos, kv_dim, kv_mul, 
                                                        head_num, head_dim, max_seq_len, nullptr);
    kernel::get_mha_kernel(base::DeviceType::DeviceCUDA)(q_cu, s_cu, k_cu, v_cu, o_cu, layer_id, pos, kv_dim, kv_mul, 
                                                        head_num, head_dim, max_seq_len, nullptr);
    o_cu.to_cpu();

    for (int32_t i = 0; i < dim; ++i) {
        ASSERT_NEAR(o_cpu.index<float>(i), o_cu.index<float>(i), 1e-5f);
    }
}

TEST(test_op_mha, mha_cuda_no_stream2) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t dim = 1024;
    int32_t kv_dim = 256;
    int32_t head_dim = 64;
    int32_t layer_num = 10;
    int32_t max_seq_len = 256;

    int32_t layer_id = 7;
    int32_t pos = 12;

    int32_t kv_mul = dim / kv_dim;
    int32_t head_num = dim / head_dim;

    tensor::Tensor q_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);
    tensor::Tensor s_cpu(base::DataType::DataTypeFp32, head_num, max_seq_len, true, allocator_cpu);
    tensor::Tensor k_cpu(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator_cpu);
    tensor::Tensor v_cpu(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator_cpu);
    tensor::Tensor o_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (int32_t i = 0; i < dim; ++i) {
        q_cpu.index<float>(i) = dist(mt);
    }
    for (int32_t l = 0; l < layer_num; ++l) {
        for (int32_t i = 0; i <= pos; ++i) {
            for (int32_t j = 0; j < kv_dim; ++j) {
                int32_t idx = l * max_seq_len * kv_dim + i * kv_dim + j;
                k_cpu.index<float>(idx) = dist(mt);
                v_cpu.index<float>(idx) = dist(mt);
            }
        }
    }
    allocator_cpu->memset_zero(o_cpu.ptr<float>(), o_cpu.byte_size(), nullptr);

    tensor::Tensor q_cu = q_cpu.clone();
    tensor::Tensor s_cu = s_cpu.clone();
    tensor::Tensor k_cu = k_cpu.clone();
    tensor::Tensor v_cu = v_cpu.clone();
    tensor::Tensor o_cu = o_cpu.clone();

    q_cu.to_cuda();
    s_cu.to_cuda();
    k_cu.to_cuda();
    v_cu.to_cuda();
    o_cu.to_cuda();

    kernel::get_mha_kernel(base::DeviceType::DeviceCPU)(q_cpu, s_cpu, k_cpu, v_cpu, o_cpu, layer_id, pos, kv_dim, kv_mul, 
                                                        head_num, head_dim, max_seq_len, nullptr);
    kernel::get_mha_kernel(base::DeviceType::DeviceCUDA)(q_cu, s_cu, k_cu, v_cu, o_cu, layer_id, pos, kv_dim, kv_mul, 
                                                        head_num, head_dim, max_seq_len, nullptr);
    o_cu.to_cpu();

    for (int32_t i = 0; i < dim; ++i) {
        ASSERT_NEAR(o_cpu.index<float>(i), o_cu.index<float>(i), 1e-5f);
    }
}

TEST(test_op_mha, mha_cuda_stream) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t dim = 512;
    int32_t kv_dim = 128;
    int32_t head_dim = 64;
    int32_t layer_num = 10;
    int32_t max_seq_len = 256;

    int32_t layer_id = 9;
    int32_t pos = 123;

    int32_t kv_mul = dim / kv_dim;
    int32_t head_num = dim / head_dim;

    tensor::Tensor q_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);
    tensor::Tensor s_cpu(base::DataType::DataTypeFp32, head_num, max_seq_len, true, allocator_cpu);
    tensor::Tensor k_cpu(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator_cpu);
    tensor::Tensor v_cpu(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator_cpu);
    tensor::Tensor o_cpu(base::DataType::DataTypeFp32, dim, true, allocator_cpu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (int32_t i = 0; i < dim; ++i) {
        q_cpu.index<float>(i) = dist(mt);
    }
    for (int32_t l = 0; l < layer_num; ++l) {
        for (int32_t i = 0; i <= pos; ++i) {
            for (int32_t j = 0; j < kv_dim; ++j) {
                int32_t idx = l * max_seq_len * kv_dim + i * kv_dim + j;
                k_cpu.index<float>(idx) = dist(mt);
                v_cpu.index<float>(idx) = dist(mt);
            }
        }
    }
    allocator_cpu->memset_zero(o_cpu.ptr<float>(), o_cpu.byte_size(), nullptr);

    tensor::Tensor q_cu = q_cpu.clone();
    tensor::Tensor s_cu = s_cpu.clone();
    tensor::Tensor k_cu = k_cpu.clone();
    tensor::Tensor v_cu = v_cpu.clone();
    tensor::Tensor o_cu = o_cpu.clone();

    q_cu.to_cuda();
    s_cu.to_cuda();
    k_cu.to_cuda();
    v_cu.to_cuda();
    o_cu.to_cuda();

    kernel::get_mha_kernel(base::DeviceType::DeviceCPU)(q_cpu, s_cpu, k_cpu, v_cpu, o_cpu, layer_id, pos, kv_dim, kv_mul, 
                                                        head_num, head_dim, max_seq_len, nullptr);
    kernel::get_mha_kernel(base::DeviceType::DeviceCUDA)(q_cu, s_cu, k_cu, v_cu, o_cu, layer_id, pos, kv_dim, kv_mul, 
                                                        head_num, head_dim, max_seq_len, nullptr);
    o_cu.to_cpu();

    for (int32_t i = 0; i < dim; ++i) {
        ASSERT_NEAR(o_cpu.index<float>(i), o_cu.index<float>(i), 1e-5f);
    }
}
