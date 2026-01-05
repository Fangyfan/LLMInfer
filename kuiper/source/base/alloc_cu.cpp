#include "base/alloc.h"
#include <cuda_runtime.h>

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::DeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) {
    if (!byte_size) {
        return nullptr;
    }

    CHECK(cudaSetDevice(3) == cudaSuccess);

    // 获取当前正在使用的 GPU 设备 ID
    int32_t cuda_id = -1;
    cudaError_t err = cudaGetDevice(&cuda_id);
    CHECK(err == cudaSuccess);
    
    // 判断本次申请的是 大块显存 (> 1MB)
    constexpr size_t mbytes = 1024 * 1024;
    if (byte_size > mbytes) {
        auto& no_busy_buffers = big_no_busy_buffers_map_[cuda_id]; // 获取当前 GPU 的大块显存池
        
        // 找到了符合条件的空闲块 (查找最小的 >= byte_size 的空闲块)
        auto it = no_busy_buffers.lower_bound(CudaMemoryBuffer(nullptr, byte_size));
        if (it != no_busy_buffers.end() && it->byte_size - byte_size < mbytes) {
            // 修改显存块为使用状态
            CudaMemoryBuffer buffer = *it;
            no_busy_buffers.erase(it);
            big_busy_buffers_map_[cuda_id].insert(buffer);
            CHECK(buffer.byte_size >= byte_size);
            // LOG(INFO) << "allocate: big reuse cuda" << cuda_id << " , ptr = " << buffer.data << " , bytes = " << byte_size << std::endl;
            return buffer.data; // 复用显存地址
        }

        // 遍历完大块显存池，没找到合适的空闲块，只能调用 cudaMalloc 申请新显存
        void* data = nullptr;
        err = cudaMalloc(&data, byte_size);
        
        // 申请失败：打印错误日志，返回空指针
        if (err != cudaSuccess) {
            char buf[256];
            snprintf(buf, sizeof(buf), 
                    "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device %d.", 
                    byte_size / mbytes, cuda_id);
            LOG(ERROR) << buf << std::endl;
            return nullptr;
        }

        // 申请成功：把这个新显存块加入到大块显存池里，标记为 busy = true（刚申请就被占用）
        // LOG(INFO) << "allocate: big malloc cuda" << cuda_id << " , ptr = " << data << " , bytes = " << byte_size << std::endl;
        big_busy_buffers_map_[cuda_id].emplace(data, byte_size);
        data_is_big[data] = true;
        return data;
    }

    auto& no_busy_buffers = cuda_no_busy_buffers_map_[cuda_id]; // 获取当前 GPU 的小块显存池

    // 找到了符合条件的空闲块 (查找最小的 >= byte_size 的空闲块)
    auto it = no_busy_buffers.lower_bound(CudaMemoryBuffer(nullptr, byte_size));
    if (it != no_busy_buffers.end()) {
        // 修改显存块为使用状态
        CudaMemoryBuffer buffer = *it;
        no_busy_buffers.erase(it);
        no_busy_bytes_count_[cuda_id] -= buffer.byte_size; // 更新小块的空闲显存
        cuda_busy_buffers_map_[cuda_id].insert(buffer);
        CHECK(buffer.byte_size >= byte_size);
        // LOG(INFO) << "allocate: small reuse cuda" << cuda_id << " , ptr = " << buffer.data << " , bytes = " << byte_size << std::endl;
        return buffer.data; // 复用显存地址
    }

    // 遍历完小块显存池，没找到合适的空闲块，只能调用 cudaMalloc 申请新显存
    void* data = nullptr;
    err = cudaMalloc(&data, byte_size);

     // 申请失败：打印错误日志，返回空指针
    if (err != cudaSuccess) {
        char buf[256];
        snprintf(buf, sizeof(buf), 
                "Error: CUDA error when allocating %lu B memory! maybe there's no enough memory left on device %d.", 
                byte_size, cuda_id);
        LOG(ERROR) << buf << std::endl;
        return nullptr;
    }

    // 申请成功：把这个新显存块加入到小块显存池里，标记为 busy = true（刚申请就被占用）
    // LOG(INFO) << "allocate: small malloc cuda" << cuda_id << " , ptr = " << data << " , bytes = " << byte_size << std::endl;
    cuda_busy_buffers_map_[cuda_id].emplace(data, byte_size);
    data_is_big[data] = false;
    return data;
}

void CUDADeviceAllocator::release(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    
    CHECK(data_is_big.count(ptr));

    if (data_is_big[ptr]) {
        // 遍历所有 GPU 的大块显存池
        for (auto& [cuda_id, busy_buffers] : big_busy_buffers_map_) {
            for (auto it = busy_buffers.begin(); it != busy_buffers.end(); it++) {
                if (it->data == ptr) { // 找到要释放的显存指针对应的显存块
                    big_busy_buffers_map_[cuda_id].insert(*it);
                    busy_buffers.erase(it);
                    // LOG(INFO) << "release: big reuse cuda" << cuda_id << " , ptr = " << ptr << std::endl;
                    return;
                }
            }
        }
    } else {
        // 遍历所有 GPU 的小块显存池
        constexpr size_t gbytes = 1024 * 1024 * 1024;
        for (auto& [cuda_id, busy_buffers] : cuda_busy_buffers_map_) {
            for (auto it = busy_buffers.begin(); it != busy_buffers.end(); it++) {
                if (it->data == ptr) { // 找到要释放的显存指针对应的显存块
                    auto& no_busy_buffers = cuda_no_busy_buffers_map_[cuda_id];
                    no_busy_bytes_count_[cuda_id] += it->byte_size; // 更新小块的空闲显存
                    no_busy_buffers.insert(*it); // 标记为空闲，完成释放
                    busy_buffers.erase(it);
                    if (no_busy_bytes_count_[cuda_id] > gbytes) { // 当前 GPU 的小块空闲显存 > 1GB
                        cudaError_t err = cudaSetDevice(cuda_id); // 切换到对应 GPU
                        CHECK(err == cudaSuccess) << "Error: CUDA error when set device " << cuda_id << std::endl;
                        for (auto& buffer : no_busy_buffers) {
                            data_is_big.erase(buffer.data);
                            err = cudaFree(buffer.data); // 调用 cudaFree 释放空闲显存
                            CHECK(err == cudaSuccess) << "Error: CUDA error when release memory on device " << cuda_id << std::endl;
                        }
                        no_busy_buffers.clear(); // 清空小块空闲显存池
                        no_busy_bytes_count_[cuda_id] = 0; // 小块的空闲显存清零
                    }
                    // LOG(INFO) << "release: small reuse cuda" << cuda_id << " , ptr = " << ptr << std::endl;
                    return;
                }
            }
        }
    }

    // // 获取当前正在使用的 GPU 设备 ID
    // int32_t cuda_id = -1;
    // cudaError_t err1 = cudaGetDevice(&cuda_id);
    // CHECK(err1 == cudaSuccess);
    // LOG(INFO) << "release: free cuda" << cuda_id << " , ptr = " << ptr << std::endl;

    // 如果不是池子里的显存，调用 cudaFree
    data_is_big.erase(ptr);
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