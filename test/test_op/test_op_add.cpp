#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernel/kernel_interface.h"
#include "../utils.cuh"

TEST(test_op_add, add_cuda_no_stream) {
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    int32_t size = 32 * 151;
    tensor::Tensor t_in1(base::DataType::DataTypeFp32, 32, 151, true, allocator_cu);
    tensor::Tensor t_in2(base::DataType::DataTypeFp32, 32, 151, true, allocator_cu);
    tensor::Tensor t_out(base::DataType::DataTypeFp32, 32, 151, true, allocator_cu);
    
    set_value_cu(t_in1.ptr<float>(), size, 2.f);
    set_value_cu(t_in2.ptr<float>(), size, 3.f);
    kernel::get_add_kernel(base::DeviceType::DeviceCUDA)(t_in1, t_in2, t_out, nullptr);
    cudaDeviceSynchronize();
    
    float* ptr = new float[size];
    cudaMemcpy(ptr, t_out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int32_t i = 0; i < size; i++) {
        ASSERT_EQ(ptr[i], 5.f);
    }
    
    delete[] ptr;
}

TEST(test_op_add, add_cuda_stream) {
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    int32_t size = 32 * 151;
    tensor::Tensor t_in1(base::DataType::DataTypeFp32, 32, 151, true, allocator_cu);
    tensor::Tensor t_in2(base::DataType::DataTypeFp32, 32, 151, true, allocator_cu);
    tensor::Tensor t_out(base::DataType::DataTypeFp32, 32, 151, true, allocator_cu);
    
    set_value_cu(t_in1.ptr<float>(), size, 2.f);
    set_value_cu(t_in2.ptr<float>(), size, 3.f);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::get_add_kernel(base::DeviceType::DeviceCUDA)(t_in1, t_in2, t_out, stream);
    cudaDeviceSynchronize();
    
    float* ptr = new float[size];
    cudaMemcpy(ptr, t_out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int32_t i = 0; i < size; i++) {
        ASSERT_EQ(ptr[i], 5.f);
    }
    
    delete[] ptr;
    cudaStreamDestroy(stream);
}

TEST(test_op_add, add_cpu) {
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t size = 32 * 151;
    tensor::Tensor t_in1(base::DataType::DataTypeFp32, 32, 151, true, allocator_cpu);
    tensor::Tensor t_in2(base::DataType::DataTypeFp32, 32, 151, true, allocator_cpu);
    tensor::Tensor t_out(base::DataType::DataTypeFp32, 32, 151, true, allocator_cpu);
    
    for (int32_t i = 0; i < size; i++) {
        t_in1.index<float>(i) = 2.f;
        t_in2.index<float>(i) = 3.f;
    }
    kernel::get_add_kernel(base::DeviceType::DeviceCPU)(t_in1, t_in2, t_out, nullptr);
    
    float* ptr = t_out.ptr<float>();
    for (int32_t i = 0; i < size; i++) {
        ASSERT_EQ(ptr[i], 5.f);
    }
}

TEST(test_op_add, add_cuda_align) {
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    int32_t size = 32 * 151 * 13;
    tensor::Tensor t_in1(base::DataType::DataTypeFp32, 32, 151, 13, true, allocator_cu);
    tensor::Tensor t_in2(base::DataType::DataTypeFp32, 32, 151, 13, true, allocator_cu);
    tensor::Tensor t_out(base::DataType::DataTypeFp32, 32, 151, 13, true, allocator_cu);
    
    set_value_cu(t_in1.ptr<float>(), size, 2.1f);
    set_value_cu(t_in2.ptr<float>(), size, 3.3f);
    kernel::get_add_kernel(base::DeviceType::DeviceCUDA)(t_in1, t_in2, t_out, nullptr);
    cudaDeviceSynchronize();

    float* ptr = new float[size];
    cudaMemcpy(ptr, t_out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int32_t i = 0; i < size; i++) {
        ASSERT_NEAR(ptr[i], 5.4f, 1e-5f);
    }

    delete[] ptr;
}