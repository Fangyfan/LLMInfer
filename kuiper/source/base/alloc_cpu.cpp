#include "base/alloc.h"

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::DeviceCPU) {}

void* CPUDeviceAllocator::allocate(size_t byte_size) {
    if (!byte_size) {
        return nullptr;
    }
    void* data = malloc(byte_size);
    return data;
}

void CPUDeviceAllocator::release(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    free(ptr);
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base