#include "base/alloc.h"
#include <cuda_runtime.h>

namespace base {
DeviceAllocator::DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

DeviceType DeviceAllocator::device_type() const {
    return device_type_;
}

void DeviceAllocator::memcpy(void* dest_ptr, const void* src_ptr, size_t byte_size, MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
    CHECK_NE(dest_ptr, nullptr);
    CHECK_NE(src_ptr, nullptr);

    if (!byte_size) {
        return;
    }
    
    // 初始化为空指针（默认流），操作是同步阻塞的（除非明确使用异步API），所有默认流的操作会相互阻塞
    cudaStream_t stream_ = nullptr;
    if (stream != nullptr) {
        // 将 void* 转换为 CUDA 流指针，操作是异步的，可以在不同流中并行执行多个拷贝/计算
        stream_ = static_cast<cudaStream_t>(stream);
    }
    
    if (memcpy_kind == MemcpyKind::MemcpyCPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, byte_size);
    } else if (memcpy_kind == MemcpyKind::MemcpyCPU2CUDA) {
        if (!stream_) { // 同步版本：阻塞直到拷贝完成
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
        } else { // 异步版本：立即返回，拷贝在流中执行
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
        }
    } else if (memcpy_kind == MemcpyKind::MemcpyCUDA2CPU) {
        if (!stream_) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
        }
    } else if (memcpy_kind == MemcpyKind::MemcpyCUDA2CUDA) {
        if (!stream_) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
        }
    } else {
        LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind) << std::endl;
    }

    if (need_sync) {
        cudaDeviceSynchronize();
    }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync) {
    CHECK(device_type_ != DeviceType::DeviceUnknown);
    
    if (device_type_ == DeviceType::DeviceCPU) {
        std::memset(ptr, 0, byte_size);
    } else {
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            cudaMemsetAsync(ptr, 0, byte_size, stream_);
        } else {
            cudaMemset(ptr, 0, byte_size);
        }
    }

    if (need_sync) {
        cudaDeviceSynchronize();
    }
}

}  // namespace base