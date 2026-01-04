#include "base/alloc.h"
#include <cuda_runtime.h>

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::DeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) {
    if (!byte_size) {
        return nullptr;
    }

    // 获取当前正在使用的 GPU 设备 ID
    int32_t cuda_id = -1;
    cudaError_t err = cudaGetDevice(&cuda_id);
    CHECK(err == cudaSuccess);
    LOG(INFO) << "allocate device = cuda" << cuda_id << std::endl;
    
    // 判断本次申请的是 大块显存 (> 1MB)
    constexpr size_t mbytes = 1024 * 1024; // 1(MB) = 1024(KB) = 1024*1024(B)
    if (byte_size > mbytes) {
        auto& big_buffers = big_buffers_map_[cuda_id]; // 获取当前 GPU 的大块显存池
        int32_t select_id = -1;
        for (int32_t i = 0; i < big_buffers.size(); i++) {
            // 遍历整个大块显存池，筛选符合条件的空闲显存块 (空间充足、空闲状态、富余空间 < 1MB)
            if (big_buffers[i].busy == false && big_buffers[i].byte_size >= byte_size &&
                big_buffers[i].byte_size - byte_size < mbytes) {
                // 如果找到了多个符合条件的，选显存最小的那个（显存浪费最少）
                if (select_id == -1 || big_buffers[select_id].byte_size > big_buffers[i].byte_size) {
                    select_id = i;
                }
            }
        }

        // 找到了符合条件的空闲块
        if (select_id != -1) {
            LOG(INFO) << "reuse big memory success" << std::endl;
            big_buffers[select_id].busy = true; // 修改显存块为使用状态
            return big_buffers[select_id].data; // 复用显存地址
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
        big_buffers.emplace_back(data, byte_size, true);
        return data;
    }

    auto& cuda_buffers = cuda_buffers_map_[cuda_id]; // 获取当前 GPU 的小块显存池
    for (int32_t i = 0; i < cuda_buffers.size(); i++) {
        // 遍历整个小块显存池，筛选符合条件的空闲显存块 (空间充足、空闲状态)
        if (cuda_buffers[i].busy == false && cuda_buffers[i].byte_size >= byte_size) {
            cuda_buffers[i].busy = true;
            no_busy_bytes_count_[cuda_id] -= cuda_buffers[i].byte_size; // 更新小块的空闲显存
            LOG(INFO) << "reuse small memory success" << std::endl;
            return cuda_buffers[i].data;
        }
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
    cuda_buffers.emplace_back(data, byte_size, true);
    return data;
}

void CUDADeviceAllocator::release(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    // 遍历所有 GPU 的大块显存池
    for (auto& [cuda_id, big_buffers] : big_buffers_map_) {
        for (int32_t i = 0; i < big_buffers.size(); i++) {
            if (big_buffers[i].data == ptr) { // 找到要释放的显存指针对应的显存块
                big_buffers[i].busy = false; // 标记为空闲，完成释放
                return;
            }
        }
    }

    // 遍历所有 GPU 的小块显存池
    constexpr size_t gbytes = 1024 * 1024 * 1024; // 1(GB) = 2^10(MB) = 2^20(KB) = 2^30(B)
    for (auto& [cuda_id, cuda_buffers] : cuda_buffers_map_) {
        for (int32_t i = 0; i < cuda_buffers.size(); i++) {
            if (cuda_buffers[i].data == ptr) { // 找到要释放的显存指针对应的显存块
                cuda_buffers[i].busy = false; // 标记为空闲，完成释放
                no_busy_bytes_count_[cuda_id] += cuda_buffers[i].byte_size; // 更新小块的空闲显存
                if (no_busy_bytes_count_[cuda_id] > gbytes) { // 当前 GPU 的小块空闲显存 > 1GB
                    std::vector<CudaMemoryBuffer> busy_buffers;
                    for (int32_t j = 0; j < cuda_buffers.size(); j++) {
                        if (cuda_buffers[j].busy) {
                            busy_buffers.push_back(cuda_buffers[j]); // 保留正在使用的显存，不能释放
                        } else {
                            cudaError_t err1 = cudaSetDevice(cuda_id); // 切换到对应 GPU
                            CHECK(err1 == cudaSuccess) << "Error: CUDA error when set device " << cuda_id << std::endl;
                            cudaError_t err2 = cudaFree(cuda_buffers[j].data); // 调用 cudaFree 释放空闲显存
                            CHECK(err2 == cudaSuccess) << "Error: CUDA error when release memory on device " << cuda_id << std::endl;
                        }
                    }
                    cuda_buffers.clear(); // 清空原有的小块显存池
                    cuda_buffers = busy_buffers; // 把正在使用的显存小块放回数组
                    no_busy_bytes_count_[cuda_id] = 0; // 小块的空闲显存清零
                }
                return;
            }
        }
    }

    // 如果不是池子里的显存，调用 cudaFree
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