#ifndef KUIPER_INCLUDE_BASE_ALLOC_H
#define KUIPER_INCLUDE_BASE_ALLOC_H

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
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}
    virtual DeviceType device_type() const { return device_type_; }
    virtual void* allocate(size_t byte_size) const = 0;
    virtual void release(void* ptr) const = 0;
    virtual void memcpy(void* dest_ptr, const void* src_ptr, size_t byte_size, MemcpyKind memcpy_kind = MemcpyKind::MemcpyCPU2CPU,
                        void* stream = nullptr, bool need_sync = false) const;
    virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);
private:
    DeviceType device_type_ = DeviceType::DeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();
    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();
    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
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