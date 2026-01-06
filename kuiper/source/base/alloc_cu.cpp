#include "base/alloc.h"
#include <cuda_runtime.h>

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::DeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) {
    if (!byte_size) {
        return nullptr;
    }

    // 确保在单卡上面分配显存，当前正在使用的 GPU 设备 ID
    CHECK(cudaSetDevice(cuda_id_) == cudaSuccess);
    
    // int32_t cuda_id = -1;
    // CHECK(cudaGetDevice(&cuda_id) == cudaSuccess);
    // LOG(INFO) << "cuda_id = " << cuda_id << std::endl;

    // 判断本次申请的是 大块显存 (> 1MB)
    constexpr size_t mbytes = 1024 * 1024;
    if (byte_size > mbytes) {
        // 找到了符合条件的空闲块 (查找最小的 >= byte_size 的空闲块，且富余空间 < 1MB)
        auto it = big_no_busy_buffers_.lower_bound(CudaMemoryBuffer(nullptr, byte_size));
        if (it != big_no_busy_buffers_.end() && it->byte_size - byte_size < mbytes) {
            void* data = it->data;
            auto pos = big_busy_buffers_.insert(*it).first; // 修改显存块为使用状态
            data_iter_[data].second = pos; // 更新 busy 显存块地址 data 与迭代器 pos 的映射关系
            big_no_busy_buffers_.erase(it);
            // LOG(INFO) << "allocate: big reuse cuda" << cuda_id_ << " , ptr = " << data << " , bytes = " << byte_size << std::endl;
            return data; // 复用显存地址
        }

        // 遍历完大块显存池，没找到合适的空闲块，只能调用 cudaMalloc 申请新显存
        void* data = nullptr;
        if (cudaMalloc(&data, byte_size) != cudaSuccess) { // 申请失败：打印错误日志，返回空指针
            char buf[256];
            snprintf(buf, sizeof(buf), 
                    "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device %d.", 
                    byte_size / mbytes, cuda_id_);
            LOG(ERROR) << buf << std::endl;
            return nullptr;
        }

        // 申请成功：把这个新显存块加入到大块显存池里，标记为 busy = true（刚申请就被占用）
        // LOG(INFO) << "allocate: big malloc cuda" << cuda_id_ << " , ptr = " << data << " , bytes = " << byte_size << std::endl;
        auto pos = big_busy_buffers_.emplace(data, byte_size).first;
        data_iter_[data] = {true, pos};
        return data;
    }

    // 找到了符合条件的空闲块 (查找最小的 >= byte_size 的空闲块)
    auto it = cuda_no_busy_buffers_.lower_bound(CudaMemoryBuffer(nullptr, byte_size));
    if (it != cuda_no_busy_buffers_.end()) {
        void* data = it->data;
        auto pos = cuda_busy_buffers_.insert(*it).first; // 修改显存块为使用状态
        data_iter_[data].second = pos; // 更新 busy 显存块中 data 指针与 pos 迭代器的映射关系
        no_busy_bytes_count_ -= it->byte_size; // 更新小块的空闲显存
        cuda_no_busy_buffers_.erase(it);
        // LOG(INFO) << "allocate: small reuse cuda" << cuda_id_ << " , ptr = " << data << " , bytes = " << byte_size << std::endl;
        return data; // 复用显存地址
    }

    // 遍历完小块显存池，没找到合适的空闲块，只能调用 cudaMalloc 申请新显存
    void* data = nullptr;
    if (cudaMalloc(&data, byte_size) != cudaSuccess) { // 申请失败：打印错误日志，返回空指针
        char buf[256];
        snprintf(buf, sizeof(buf), 
                "Error: CUDA error when allocating %lu B memory! maybe there's no enough memory left on device %d.", 
                byte_size, cuda_id_);
        LOG(ERROR) << buf << std::endl;
        return nullptr;
    }

    // 申请成功：把这个新显存块加入到小块显存池里，标记为 busy = true（刚申请就被占用）
    // LOG(INFO) << "allocate: small malloc cuda" << cuda_id_ << " , ptr = " << data << " , bytes = " << byte_size << std::endl;
    auto pos = cuda_busy_buffers_.emplace(data, byte_size).first;
    data_iter_[data] = {false, pos};
    return data;
}

void CUDADeviceAllocator::release(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    
    auto iter = data_iter_.find(ptr);
    CHECK(iter != data_iter_.end());
    auto [data_is_big, it] = iter->second; // 找到要释放的显存指针对应的显存块迭代器 it

    if (data_is_big) {
        big_no_busy_buffers_.insert(*it);
        big_busy_buffers_.erase(it);
        // LOG(INFO) << "released: big reuse cuda" << cuda_id_ << " , ptr = " << ptr << std::endl;
        return;
    } else {
        cuda_no_busy_buffers_.insert(*it); // 标记为空闲，完成释放
        no_busy_bytes_count_ += it->byte_size; // 更新小块的空闲显存
        cuda_busy_buffers_.erase(it);
        // LOG(INFO) << "released: small reuse cuda" << cuda_id_ << " , ptr = " << ptr << std::endl;
        constexpr size_t gbytes = 1024 * 1024 * 1024;
        if (no_busy_bytes_count_ > gbytes) { // 当前 GPU 的小块空闲显存 > 1024MB
            // LOG(INFO) << "released: free all small blocks cuda" << cuda_id_ << std::endl;
            for (auto& buffer : cuda_no_busy_buffers_) {
                data_iter_.erase(buffer.data);
                cudaError_t err = cudaFree(buffer.data); // 调用 cudaFree 释放空闲显存
                CHECK(err == cudaSuccess) << "Error: CUDA error when release memory on device " << cuda_id_ << std::endl;
            }
            cuda_no_busy_buffers_.clear(); // 清空小块空闲显存池
            no_busy_bytes_count_ = 0; // 小块的空闲显存清零
        }
        return;
    }

    // 如果不是池子里的显存，调用 cudaFree
    // LOG(INFO) << "released: free cuda" << cuda_id_ << " , ptr = " << ptr << std::endl;
    data_iter_.erase(ptr);
    cudaError_t err = cudaFree(ptr);
    CHECK(err == cudaSuccess) << "Error: CUDA error when release memory on device" << std::endl;
}

// void* CUDADeviceAllocator::allocate(size_t byte_size) {
//     if (!byte_size) {
//         return nullptr;
//     }
//     void* ptr = nullptr;
//     cudaError_t err = cudaMalloc(&ptr, byte_size);
//     CHECK_EQ(err, cudaSuccess);
//     return ptr;
// }

// void CUDADeviceAllocator::release(void* ptr) {
//     if (ptr == nullptr) {
//         return;
//     }
//     cudaFree(ptr);
// }

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;
}  // namespace base