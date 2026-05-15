#include <random>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "sampler/argmax_sampler.h"
#include "tensor/tensor.h"
#include "base/cuda_config.h"

TEST(test_op_argmax, argmax_cuda_no_stream1) {
    int32_t size = 151936;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor token_cu(base::DataType::DataTypeInt32, 1, true, allocator_cu);
    tensor::Tensor buffer_cu(base::DataType::DataTypeInt32, 128 * 2, true, allocator_cu);

    t_cpu.index<float>(0) = 0;
    t_cpu.index<float>(1) = 1;
    t_cpu.index<float>(2) = 2;
    for (int32_t i = 3; i < 15; ++i) {
        t_cpu.index<float>(i) = 3;
    }
    for (int32_t i = 15; i < size; ++i) {
        t_cpu.index<float>(i) = 2.9;
    }

    tensor::Tensor t_cu = t_cpu.clone();
    t_cu.to_cuda();

    auto sampler_cpu = std::make_unique<sampler::ArgmaxSampler>(base::DeviceType::DeviceCPU);
    auto sampler_cu = std::make_unique<sampler::ArgmaxSampler>(base::DeviceType::DeviceCUDA);

    int32_t x = sampler_cpu->sample(t_cpu.ptr<float>(), t_cpu.size(), token_cu.ptr<int32_t>(), buffer_cu.ptr<void>(), nullptr);
    int32_t y = sampler_cu->sample(t_cu.ptr<float>(), t_cu.size(), token_cu.ptr<int32_t>(), buffer_cu.ptr<void>(), nullptr);

    ASSERT_EQ(x, y);
    ASSERT_EQ(x, 3);
    ASSERT_EQ(y, 3);
}

TEST(test_op_argmax, argmax_cuda_no_stream2) {
    int32_t size = 151936;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor token_cu(base::DataType::DataTypeInt32, 1, true, allocator_cu);
    tensor::Tensor buffer_cu(base::DataType::DataTypeInt32, 128 * 2, true, allocator_cu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
        t_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor t_cu = t_cpu.clone();
    t_cu.to_cuda();

    auto sampler_cpu = std::make_unique<sampler::ArgmaxSampler>(base::DeviceType::DeviceCPU);
    auto sampler_cu = std::make_unique<sampler::ArgmaxSampler>(base::DeviceType::DeviceCUDA);

    int32_t x = sampler_cpu->sample(t_cpu.ptr<float>(), t_cpu.size(), token_cu.ptr<int32_t>(), buffer_cu.ptr<void>(), nullptr);
    int32_t y = sampler_cu->sample(t_cu.ptr<float>(), t_cu.size(), token_cu.ptr<int32_t>(), buffer_cu.ptr<void>(), nullptr);

    ASSERT_EQ(x, y);
}

TEST(test_op_argmax, argmax_cuda_stream1) {
    int32_t size = 151936;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor token_cu(base::DataType::DataTypeInt32, 1, true, allocator_cu);
    tensor::Tensor buffer_cu(base::DataType::DataTypeInt32, 128 * 2, true, allocator_cu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
        t_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor t_cu = t_cpu.clone();
    t_cu.to_cuda();

    auto sampler_cpu = std::make_unique<sampler::ArgmaxSampler>(base::DeviceType::DeviceCPU);
    auto sampler_cu = std::make_unique<sampler::ArgmaxSampler>(base::DeviceType::DeviceCUDA);

    auto cuda_config = std::make_unique<kernel::CudaConfig>();
    cuda_config->create();
    ASSERT_NE(cuda_config->stream, nullptr);

    int32_t x = sampler_cpu->sample(t_cpu.ptr<float>(), t_cpu.size(), token_cu.ptr<int32_t>(), buffer_cu.ptr<void>(), nullptr);
    int32_t y = sampler_cu->sample(t_cu.ptr<float>(), t_cu.size(), token_cu.ptr<int32_t>(), buffer_cu.ptr<void>(), cuda_config->stream);

    ASSERT_EQ(x, y);
}

TEST(test_op_argmax, argmax_cuda_stream2) {
    int32_t size = 151936;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cpu(base::DataType::DataTypeFp32, size, true, allocator_cpu);
    tensor::Tensor token_cu(base::DataType::DataTypeInt32, 1, true, allocator_cu);
    tensor::Tensor buffer_cu(base::DataType::DataTypeInt32, 128 * 2, true, allocator_cu);

    std::mt19937 mt(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < size; ++i) {
        t_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor t_cu = t_cpu.clone();
    t_cu.to_cuda();

    auto sampler_cpu = std::make_unique<sampler::ArgmaxSampler>(base::DeviceType::DeviceCPU);
    auto sampler_cu = std::make_unique<sampler::ArgmaxSampler>(base::DeviceType::DeviceCUDA);

    auto cuda_config = std::make_unique<kernel::CudaConfig>();
    cuda_config->create();
    ASSERT_NE(cuda_config->stream, nullptr);

    int32_t x = sampler_cpu->sample(t_cpu.ptr<float>(), t_cpu.size(), token_cu.ptr<int32_t>(), buffer_cu.ptr<void>(), nullptr);
    int32_t y = sampler_cu->sample(t_cu.ptr<float>(), t_cu.size(), token_cu.ptr<int32_t>(), buffer_cu.ptr<void>(), cuda_config->stream);

    ASSERT_EQ(x, y);
}