#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"
#include "../utils.cuh"

TEST(test_buffer, allocate) {
    using namespace base;
    auto allocator = CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(32, allocator);
    // LOG(INFO) << "buffer allocate ptr: " << buffer.ptr() << std::endl;
    ASSERT_NE(buffer.ptr(), nullptr);
    ASSERT_EQ(buffer.byte_size(), 32);
    ASSERT_EQ(buffer.is_external(), false);
    ASSERT_EQ(buffer.allocator(), allocator);
    ASSERT_EQ(buffer.device_type(), DeviceType::DeviceCPU);
    buffer.set_device_type(DeviceType::DeviceCUDA);
    ASSERT_EQ(buffer.device_type(), DeviceType::DeviceCUDA);

    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    Buffer buffer_cu(32, allocator_cu);
    // LOG(INFO) << "buffer_cu allocate ptr: " << buffer_cu.ptr() << std::endl;
    ASSERT_NE(buffer_cu.ptr(), nullptr);
    ASSERT_EQ(buffer_cu.byte_size(), 32);
    ASSERT_EQ(buffer_cu.is_external(), false);
    ASSERT_EQ(buffer_cu.allocator(), allocator_cu);
    ASSERT_EQ(buffer_cu.device_type(), DeviceType::DeviceCUDA);
}

TEST(test_buffer, use_external) {
    using namespace base;
    auto allocator = CPUDeviceAllocatorFactory::get_instance();
    float* ptr = new float[32];
    Buffer buffer(32, nullptr, ptr, true);
    ASSERT_EQ(buffer.is_external(), true);
    ASSERT_EQ(buffer.byte_size(), 32);
    ASSERT_EQ(buffer.ptr(), ptr);
}

TEST(test_buffer, cuda_memory1) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    
    int32_t size = 32;
    float* ptr = new float[size];
    for (int32_t i = 0; i < size; i++) {
        ptr[i] = float(i);
    }

    Buffer buffer(size * sizeof(float), nullptr, ptr, true);
    buffer.set_device_type(DeviceType::DeviceCPU);
    ASSERT_EQ(buffer.is_external(), true);

    Buffer buffer_cu(size * sizeof(float), allocator_cu);
    buffer_cu.copy_from(buffer);

    float* ptr2 = new float[size];
    // LOG(INFO) << "copy buffer_cu.ptr() " << buffer_cu.ptr() << " to ptr2 " << ptr2 << ", byte_size = " << size * sizeof(float) << std::endl;
    allocator_cu->memcpy(ptr2, buffer_cu.ptr(), size * sizeof(float), MemcpyKind::MemcpyCUDA2CPU);
    // cudaMemcpy(ptr2, buffer_cu.ptr(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int32_t i = 0; i < size; i++) {
        ASSERT_EQ(ptr2[i], ptr[i]);
    }

    delete[] ptr;
    delete[] ptr2;
}

TEST(test_buffer, cuda_memory2) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32;
    Buffer buffer_cu1(size * sizeof(float), allocator_cu);
    Buffer buffer_cu2(size * sizeof(float), allocator_cu);
    ASSERT_EQ(buffer_cu1.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(buffer_cu2.device_type(), DeviceType::DeviceCUDA);

    set_value_cu(static_cast<float*>(buffer_cu1.ptr()), size, 1.f);

    buffer_cu2.copy_from(buffer_cu1);

    float* ptr = new float[size];
    // allocator_cu->memcpy(ptr, buffer_cu2.ptr(), size * sizeof(float), MemcpyKind::MemcpyCUDA2CPU);
    cudaMemcpy(ptr, buffer_cu2.ptr(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int32_t i = 0; i < size; i++) {
        ASSERT_EQ(ptr[i], 1.f);
    }

    delete[] ptr;
}

TEST(test_buffer, cuda_memory3) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    auto allocator = CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32;
    Buffer buffer_cu(size * sizeof(float), allocator_cu);
    Buffer buffer(size * sizeof(float), allocator);
    ASSERT_EQ(buffer_cu.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(buffer.device_type(), DeviceType::DeviceCPU);

    set_value_cu(static_cast<float*>(buffer_cu.ptr()), size, 1.f);

    buffer.copy_from(buffer_cu);

    float* ptr = static_cast<float*>(buffer.ptr());
    for (int32_t i = 0; i < size; i++) {
        ASSERT_EQ(ptr[i], 1.f);
    }

    // delete[] ptr; 这里不要手动释放 ptr，因为 ptr 由 buffer 管理，会重复释放
}