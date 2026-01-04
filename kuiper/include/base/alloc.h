#ifndef KUIPER_INCLUDE_BASE_ALLOC_H
#define KUIPER_INCLUDE_BASE_ALLOC_H

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
    bool busy;
    CudaMemoryBuffer() = default;
    CudaMemoryBuffer(void* data, size_t byte_size, bool busy) : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();
    void* allocate(size_t byte_size) override;
    void release(void* ptr) override;
private:
    std::map<int32_t, size_t> no_busy_bytes_count_;
    std::map<int32_t, std::vector<CudaMemoryBuffer>> big_buffers_map_;
    std::map<int32_t, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
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