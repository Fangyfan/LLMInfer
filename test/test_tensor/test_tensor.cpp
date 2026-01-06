#include <cuda_runtime_api.h>
#include <tensor/tensor.h>
#include <gtest/gtest.h>
#include "../utils.cuh"

TEST(test_tensor, tensor_base) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cu(DataType::DataTypeFp32, 32, 32, true, allocator_cu);

    set_value_cu(t_cu.ptr<float>(), t_cu.size(), 1.f);
    t_cu.to_cpu();

    ASSERT_EQ(t_cu.device_type(), DeviceType::DeviceCPU);
    float* ptr_cpu = t_cu.ptr<float>();
    for (int32_t i = 0; i < t_cu.size(); i++) {
        ASSERT_EQ(ptr_cpu[i], 1.f);
    }
}

TEST(test_tensor, clone_cuda) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cu(DataType::DataTypeFp32, 32, 32, true, allocator_cu);
    set_value_cu(t1_cu.ptr<float>(), t1_cu.size(), 1.f);

    tensor::Tensor t2_cu = t1_cu.clone();
    ASSERT_EQ(t2_cu.is_empty(), false);
    ASSERT_EQ(t2_cu.ptr<void>(), t2_cu.get_buffer()->ptr());
    ASSERT_EQ(t2_cu.get_dim(0), 32);
    ASSERT_EQ(t2_cu.get_dim(1), 32);
    ASSERT_EQ(t2_cu.size(), 32 * 32);
    ASSERT_EQ(t2_cu.dims(), std::vector<int32_t>({32, 32}));
    ASSERT_EQ(t2_cu.dims_size(), 2);
    ASSERT_EQ(t2_cu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t2_cu.strides(), std::vector<size_t>({32, 1}));
    ASSERT_EQ(t2_cu.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(t2_cu.byte_size(), 32 * 32 * data_type_size(DataType::DataTypeFp32));

    float* p2 = new float[32 * 32];
    allocator_cu->memcpy(p2, t2_cu.ptr<float>(), t2_cu.size() * sizeof(float), MemcpyKind::MemcpyCUDA2CPU);
    for (int32_t i = 0; i < t2_cu.size(); i++) {
        ASSERT_EQ(p2[i], 1.f);
    }

    allocator_cu->memcpy(p2, t1_cu.ptr<float>(), t1_cu.size() * sizeof(float), MemcpyKind::MemcpyCUDA2CPU);
    for (int32_t i = 0; i < t1_cu.size(); i++) {
        ASSERT_EQ(p2[i], 1.f);
    }

    t2_cu.to_cpu();
    std::memcpy(p2, t2_cu.ptr<float>(), t2_cu.size() * sizeof(float));
    for (int32_t i = 0; i < t2_cu.size(); i++) {
        ASSERT_EQ(p2[i], 1.f);
    }

    delete[] p2;
}

TEST(test_tensor, clone_cpu) {
    using namespace base;
    auto allocator = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(DataType::DataTypeFp32, 32, 32, true, allocator);
    for (int32_t i = 0; i < t1_cpu.size(); i++) {
        t1_cpu.index<float>(i) = 1.f;
    }
    
    tensor::Tensor t2_cpu = t1_cpu.clone();
    float* p2 = new float[32 * 32];
    std::memcpy(p2, t2_cpu.ptr<float>(), 32 * 32 * sizeof(float));
    for (int32_t i = 0; i < t2_cpu.size(); i++) {
        ASSERT_EQ(p2[i], 1.f);
    }
    std::memcpy(p2, t1_cpu.ptr<float>(), 32 * 32 * sizeof(float));
    for (int32_t i = 0; i < t1_cpu.size(); i++) {
        ASSERT_EQ(p2[i], 1.f);
    }

    delete[] p2;
}

TEST(test_tensor, to_cuda) {
    using namespace base;
    auto allocator_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cpu(DataType::DataTypeFp32, 32, 32, true, allocator_cpu);
    float* p1 = t_cpu.ptr<float>();
    for (int32_t i = 0; i < t_cpu.size(); i++) {
        *(p1 + i) = 1.f;
    }
    
    t_cpu.to_cuda();
    
    float* p2 = new float[32 * 32];
    allocator_cpu->memcpy(p2, t_cpu.ptr<float>(), 32 * 32 * sizeof(float), MemcpyKind::MemcpyCUDA2CPU);
    for (int32_t i = 0; i < t_cpu.size(); i++) {
        ASSERT_EQ(p2[i], 1.f);
    }

    delete[] p2;
}

TEST(test_tensor, init1) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cu(DataType::DataTypeFp32, 32, true, allocator_cu);
    ASSERT_EQ(t_cu.is_empty(), false);
    ASSERT_EQ(t_cu.get_dim(0), 32);
    ASSERT_EQ(t_cu.size(), 32);
    ASSERT_EQ(t_cu.dims(), std::vector<int32_t>({32}));
    ASSERT_EQ(t_cu.dims_size(), 1);
    ASSERT_EQ(t_cu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cu.strides(), std::vector<size_t>({1}));
    ASSERT_EQ(t_cu.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(t_cu.byte_size(), 32 * data_type_size(DataType::DataTypeFp32));
}

TEST(test_tensor, init2) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cu(DataType::DataTypeFp32, 32, 32, true, allocator_cu);
    ASSERT_EQ(t_cu.is_empty(), false);
    ASSERT_EQ(t_cu.get_dim(0), 32);
    ASSERT_EQ(t_cu.get_dim(1), 32);
    ASSERT_EQ(t_cu.size(), 32 * 32);
    ASSERT_EQ(t_cu.dims(), std::vector<int32_t>({32, 32}));
    ASSERT_EQ(t_cu.dims_size(), 2);
    ASSERT_EQ(t_cu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cu.strides(), std::vector<size_t>({32, 1}));
    ASSERT_EQ(t_cu.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(t_cu.byte_size(), 32 * 32 * data_type_size(DataType::DataTypeFp32));
}

TEST(test_tensor, init3) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cu(DataType::DataTypeFp32, 12, 22, 32, true, allocator_cu);
    ASSERT_EQ(t_cu.is_empty(), false);
    ASSERT_EQ(t_cu.get_dim(0), 12);
    ASSERT_EQ(t_cu.get_dim(1), 22);
    ASSERT_EQ(t_cu.get_dim(2), 32);
    ASSERT_EQ(t_cu.size(), 12 * 22 * 32);
    ASSERT_EQ(t_cu.dims(), std::vector<int32_t>({12, 22, 32}));
    ASSERT_EQ(t_cu.dims_size(), 3);
    ASSERT_EQ(t_cu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cu.strides(), std::vector<size_t>({22 * 32, 32, 1}));
    ASSERT_EQ(t_cu.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(t_cu.byte_size(), 12 * 22 * 32 * data_type_size(DataType::DataTypeFp32));
}

TEST(test_tensor, init5) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cu(DataType::DataTypeFp32, {12, 22, 32, 42, 52}, true, allocator_cu);
    ASSERT_EQ(t_cu.is_empty(), false);
    ASSERT_EQ(t_cu.get_dim(0), 12);
    ASSERT_EQ(t_cu.get_dim(1), 22);
    ASSERT_EQ(t_cu.get_dim(2), 32);
    ASSERT_EQ(t_cu.get_dim(3), 42);
    ASSERT_EQ(t_cu.get_dim(4), 52);
    ASSERT_EQ(t_cu.size(), 12 * 22 * 32 * 42 * 52);
    ASSERT_EQ(t_cu.dims(), std::vector<int32_t>({12, 22, 32, 42, 52}));
    ASSERT_EQ(t_cu.dims_size(), 5);
    ASSERT_EQ(t_cu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cu.strides(), std::vector<size_t>({22 * 32 * 42 * 52, 32 * 42 * 52, 42 * 52, 52, 1}));
    ASSERT_EQ(t_cu.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(t_cu.byte_size(), 12 * 22 * 32 * 42 * 52 * data_type_size(DataType::DataTypeFp32));
}

TEST(test_tensor, init4) {
    using namespace base;
    auto allocator_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cu(DataType::DataTypeFp32, 12, 22, 32, 42, true, allocator_cu);
    ASSERT_EQ(t_cu.is_empty(), false);
    ASSERT_EQ(t_cu.get_dim(0), 12);
    ASSERT_EQ(t_cu.get_dim(1), 22);
    ASSERT_EQ(t_cu.get_dim(2), 32);
    ASSERT_EQ(t_cu.get_dim(3), 42);
    ASSERT_EQ(t_cu.size(), 12 * 22 * 32 * 42);
    ASSERT_EQ(t_cu.dims(), std::vector<int32_t>({12, 22, 32, 42}));
    ASSERT_EQ(t_cu.dims_size(), 4);
    ASSERT_EQ(t_cu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cu.strides(), std::vector<size_t>({22 * 32 * 42, 32 * 42, 42, 1}));
    ASSERT_EQ(t_cu.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(t_cu.byte_size(), 12 * 22 * 32 * 42 * data_type_size(DataType::DataTypeFp32));
}

TEST(test_tensor, init6) {
    using namespace base;
    float* ptr = new float[32];
    ptr[0] = 31;
    ptr[1] = 32;
    tensor::Tensor t_cpu(base::DataType::DataTypeFp32, 32, false, nullptr, ptr);
    t_cpu.set_device_type(DeviceType::DeviceCPU);
    ASSERT_EQ(t_cpu.is_empty(), false);
    ASSERT_EQ(t_cpu.get_dim(0), 32);
    ASSERT_EQ(t_cpu.size(), 32);
    ASSERT_EQ(t_cpu.dims(), std::vector<int32_t>({32}));
    ASSERT_EQ(t_cpu.dims_size(), 1);
    ASSERT_EQ(t_cpu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cpu.strides(), std::vector<size_t>({1}));
    ASSERT_EQ(t_cpu.device_type(), DeviceType::DeviceCPU);
    ASSERT_EQ(t_cpu.byte_size(), 32 * data_type_size(DataType::DataTypeFp32));
    ASSERT_EQ(t_cpu.ptr<float>(), ptr);
    ASSERT_EQ(*t_cpu.ptr<float>(), 31);
    ASSERT_EQ(*t_cpu.ptr<float>(0), 31);
    ASSERT_EQ(*t_cpu.ptr<float>(1), 32);
    ASSERT_EQ(t_cpu.index<float>(1), 32);

    t_cpu.to_cuda();
    set_value_cu(t_cpu.ptr<float>(), t_cpu.size(), 2.f);
    ASSERT_EQ(t_cpu.is_empty(), false);
    ASSERT_EQ(t_cpu.get_dim(0), 32);
    ASSERT_EQ(t_cpu.size(), 32);
    ASSERT_EQ(t_cpu.dims(), std::vector<int32_t>({32}));
    ASSERT_EQ(t_cpu.dims_size(), 1);
    ASSERT_EQ(t_cpu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cpu.strides(), std::vector<size_t>({1}));
    ASSERT_EQ(t_cpu.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(t_cpu.byte_size(), 32 * data_type_size(DataType::DataTypeFp32));

    float* p2 = new float[32];
    cudaMemcpy(p2, t_cpu.ptr<float>(), t_cpu.byte_size(), cudaMemcpyDeviceToHost);
    for (int32_t i = 0; i < 32; i++) {
        ASSERT_EQ(p2[i], 2.f);
    }
    delete[] p2;
}

TEST(test_tensor, init7) {
    using namespace base;
    float* ptr = new float[32];
    ptr[0] = 31;
    ptr[1] = 32;
    tensor::Tensor t_cpu(base::DataType::DataTypeFp32, 32);
    auto buffer = std::make_shared<base::Buffer>(t_cpu.byte_size(), nullptr, ptr, true);
    buffer->set_device_type(DeviceType::DeviceCPU); // 当没有 allocator 来构造 buffer 时，需要手动设置 device_type
    ASSERT_EQ(t_cpu.assgin(buffer), true); // 将 buffer 赋值给 t_cpu

    ASSERT_EQ(t_cpu.is_empty(), false);
    ASSERT_EQ(t_cpu.get_dim(0), 32);
    ASSERT_EQ(t_cpu.size(), 32);
    ASSERT_EQ(t_cpu.dims(), std::vector<int32_t>({32}));
    ASSERT_EQ(t_cpu.dims_size(), 1);
    ASSERT_EQ(t_cpu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cpu.strides(), std::vector<size_t>({1}));
    ASSERT_EQ(t_cpu.device_type(), DeviceType::DeviceCPU);
    ASSERT_EQ(t_cpu.byte_size(), 32 * data_type_size(DataType::DataTypeFp32));
    ASSERT_EQ(t_cpu.ptr<float>(), ptr);
    ASSERT_EQ(*t_cpu.ptr<float>(), 31);
    ASSERT_EQ(*t_cpu.ptr<float>(0), 31);
    ASSERT_EQ(*t_cpu.ptr<float>(1), 32);
    ASSERT_EQ(t_cpu.index<float>(1), 32);

    t_cpu.to_cuda();
    set_value_cu(t_cpu.ptr<float>(), t_cpu.size(), 2.f);
    ASSERT_EQ(t_cpu.is_empty(), false);
    ASSERT_EQ(t_cpu.get_dim(0), 32);
    ASSERT_EQ(t_cpu.size(), 32);
    ASSERT_EQ(t_cpu.dims(), std::vector<int32_t>({32}));
    ASSERT_EQ(t_cpu.dims_size(), 1);
    ASSERT_EQ(t_cpu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cpu.strides(), std::vector<size_t>({1}));
    ASSERT_EQ(t_cpu.device_type(), DeviceType::DeviceCUDA);
    ASSERT_EQ(t_cpu.byte_size(), 32 * data_type_size(DataType::DataTypeFp32));

    float* p2 = new float[32];
    cudaMemcpy(p2, t_cpu.ptr<float>(), t_cpu.byte_size(), cudaMemcpyDeviceToHost);
    for (int32_t i = 0; i < 32; i++) {
        ASSERT_EQ(p2[i], 2.f);
    }
    delete[] p2;
}

// TEST(test_tensor, index) {
//     using namespace base;
//     float* ptr = new float[32];
//     ptr[0] = 31;
//     tensor::Tensor t_cpu(base::DataType::DataTypeFp32, 32, false, nullptr, ptr);
//     void* vp = t_cpu.ptr<void>();
//     vp += 1; // 地址 + 1
    
//     float* fp = t_cpu.ptr<float>();
//     ASSERT_EQ(*fp, 31);
//     fp += 1; // 地址 + 4
    
//     LOG(INFO) << "t.ptr = " << t_cpu.ptr<void>() << ", vp = " << vp << ", fp = " << (void*)fp << "\n";
//     ASSERT_NE(vp, fp);
//     delete[] ptr;
// }

TEST(test_tensor, dims_stride) {
    using namespace base;
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    tensor::Tensor t_cpu(base::DataType::DataTypeFp32, 32, 32, 3, true, allocator_cpu);
    ASSERT_EQ(t_cpu.is_empty(), false);
    ASSERT_EQ(t_cpu.get_dim(0), 32);
    ASSERT_EQ(t_cpu.get_dim(1), 32);
    ASSERT_EQ(t_cpu.get_dim(2), 3);

    // const 引用绑定临时对象（安全，生命周期延长）
    // 直接复用返回的临时对象，零拷贝开销
    const auto& strides = t_cpu.strides();

    // 非 const 引用绑定临时对象（编译报错！）
    // 用普通引用绑定，会导致 “悬空引用”（引用指向已销毁的对象）
    // auto& strides2 = get_strides(); // 错误：无法将临时对象绑定到非 const 左值引用

    ASSERT_EQ(strides.at(0), 32 * 3);
    ASSERT_EQ(strides.at(1), 3);
    ASSERT_EQ(strides.at(2), 1);
}

TEST(test_tensor, assign) {
    using namespace base;
    auto allocator_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t_cpu(DataType::DataTypeFp32, 32, 32, true, allocator_cpu);
    
    int32_t size = 32 * 32;
    float* ptr = new float[size];
    for (int32_t i = 0; i < size; i++) {
        ptr[i] = float(i);
    }

    auto buffer = std::make_shared<base::Buffer>(size * sizeof(float), nullptr, ptr, true);
    buffer->set_device_type(DeviceType::DeviceCPU); // 当没有 allocator 来构造 buffer 时，需要手动设置 device_type
    ASSERT_EQ(t_cpu.assgin(buffer), true); // 将 buffer 赋值给 t_cpu

    ASSERT_EQ(t_cpu.is_empty(), false);
    ASSERT_EQ(t_cpu.get_dim(0), 32);
    ASSERT_EQ(t_cpu.get_dim(1), 32);
    ASSERT_EQ(t_cpu.size(), 32 * 32);
    ASSERT_EQ(t_cpu.dims(), std::vector<int32_t>({32, 32}));
    ASSERT_EQ(t_cpu.dims_size(), 2);
    ASSERT_EQ(t_cpu.data_type(), DataType::DataTypeFp32);
    ASSERT_EQ(t_cpu.strides(), std::vector<size_t>({32, 1}));
    ASSERT_EQ(t_cpu.device_type(), DeviceType::DeviceCPU);
    ASSERT_EQ(t_cpu.byte_size(), 32 * 32 * data_type_size(DataType::DataTypeFp32));
    ASSERT_EQ(*t_cpu.ptr<float>(), 0.f);
    ASSERT_EQ(*t_cpu.ptr<float>(0), 0.f);
    ASSERT_EQ(*t_cpu.ptr<float>(1), 1.f);
    ASSERT_EQ(*t_cpu.ptr<float>(2), 2.f);
    ASSERT_EQ(t_cpu.index<float>(31), 31.f);

    delete[] ptr;
}