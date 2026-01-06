#ifndef KUIPER_INCLUDE_BASE_ALLOC_H
#define KUIPER_INCLUDE_BASE_ALLOC_H

#include <set>
#include <map>
#include <memory>
#include "base/base.h"

namespace base {
enum class MemcpyKind : uint8_t {
    MemcpyCPU2CPU = 0,
    MemcpyCPU2CUDA = 1,
    MemcpyCUDA2CPU = 2,
    MemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type);
    virtual DeviceType device_type() const;
    virtual void* allocate(size_t byte_size) = 0; // 要实现的内存分配接口
    virtual void release(void* ptr) = 0; // 要实现的内存释放接口
    virtual void memcpy(void* dest_ptr, const void* src_ptr, size_t byte_size, MemcpyKind memcpy_kind = MemcpyKind::MemcpyCPU2CPU,
                        void* stream = nullptr, bool need_sync = false) const;
    virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);
private:
    DeviceType device_type_ = DeviceType::DeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();
    void* allocate(size_t byte_size) override;
    void release(void* ptr) override;
};

struct CudaMemoryBuffer {
    void* data;
    size_t byte_size;
    CudaMemoryBuffer() = default;
    CudaMemoryBuffer(void* data, size_t byte_size) : data(data), byte_size(byte_size) {}
    bool operator < (const CudaMemoryBuffer& buffer) const {
        if (byte_size == buffer.byte_size) return data < buffer.data;
        return byte_size < buffer.byte_size;
    }
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();
    void* allocate(size_t byte_size) override;
    void release(void* ptr) override;
private:
    const int32_t cuda_id_ = 3; // 确保在单卡上面分配显存，当前正在使用的 GPU 设备 ID
    size_t no_busy_bytes_count_ = 0; // 记录 空闲的 显存小块 大小之和
    std::set<CudaMemoryBuffer> big_busy_buffers_;
    std::set<CudaMemoryBuffer> big_no_busy_buffers_;
    std::set<CudaMemoryBuffer> cuda_busy_buffers_;
    std::set<CudaMemoryBuffer> cuda_no_busy_buffers_;
    std::map<void*, std::pair<bool, std::set<CudaMemoryBuffer>::iterator>> data_iter_; // 记录 busy 显存块 ptr -> iter 映射关系
};

class CPUDeviceAllocatorFactory {
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }
private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CUDADeviceAllocator>();
        }
        return instance;
    }
private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};
}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_ALLOC_H