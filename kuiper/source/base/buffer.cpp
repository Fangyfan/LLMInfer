#include "base/buffer.h"

namespace base {
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr, bool use_external)
: byte_size_(byte_size), allocator_(allocator), ptr_(ptr), use_external_(use_external) {
    if (!ptr_ && allocator_) {
        device_type_ = allocator_->device_type();
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
    }
}

Buffer::~Buffer() {
    if (!use_external_) { // 当 use_external = false 时，表示当前 Buffer 拥有该内存，即这块资源需要 Buffer 进行管理
        if (ptr_ && allocator_) {
            allocator_->release(ptr_); // 自动释放这块内存
            ptr_ = nullptr;
        }
    }
}

bool Buffer::allocate() {
    if (allocator_ && byte_size_) {
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
        return ptr_ != nullptr;
    }
    return false;
}

void Buffer::copy_from(const Buffer& buffer) const {
    CHECK(allocator_ != nullptr);
    CHECK(buffer.ptr_ != nullptr);

    size_t copy_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
    
    DeviceType dest_device = device_type_;
    DeviceType src_device = buffer.device_type_;
    CHECK(src_device != DeviceType::DeviceUnknown && dest_device != DeviceType::DeviceUnknown);

    if (src_device == DeviceType::DeviceCPU && dest_device == DeviceType::DeviceCPU) {
        allocator_->memcpy(ptr_, buffer.ptr_, copy_size, MemcpyKind::MemcpyCPU2CPU);
    } else if (src_device == DeviceType::DeviceCPU && dest_device == DeviceType::DeviceCUDA) {
        allocator_->memcpy(ptr_, buffer.ptr_, copy_size, MemcpyKind::MemcpyCPU2CUDA);
    } else if (src_device == DeviceType::DeviceCUDA && dest_device == DeviceType::DeviceCPU) {
        allocator_->memcpy(ptr_, buffer.ptr_, copy_size, MemcpyKind::MemcpyCUDA2CPU);
    } else {
        allocator_->memcpy(ptr_, buffer.ptr_, copy_size, MemcpyKind::MemcpyCUDA2CUDA);
    }
}

void Buffer::copy_from(const Buffer* buffer) const {
    CHECK(allocator_ != nullptr);
    CHECK(buffer != nullptr && buffer->ptr_ != nullptr);

    size_t copy_size = byte_size_ < buffer->byte_size_ ? byte_size_ : buffer->byte_size_;
    
    DeviceType dest_device = device_type_;
    DeviceType src_device = buffer->device_type_;
    CHECK(src_device != DeviceType::DeviceUnknown && dest_device != DeviceType::DeviceUnknown);

    if (src_device == DeviceType::DeviceCPU && dest_device == DeviceType::DeviceCPU) {
        allocator_->memcpy(ptr_, buffer->ptr_, copy_size, MemcpyKind::MemcpyCPU2CPU);
    } else if (src_device == DeviceType::DeviceCPU && dest_device == DeviceType::DeviceCUDA) {
        allocator_->memcpy(ptr_, buffer->ptr_, copy_size, MemcpyKind::MemcpyCPU2CUDA);
    } else if (src_device == DeviceType::DeviceCUDA && dest_device == DeviceType::DeviceCPU) {
        allocator_->memcpy(ptr_, buffer->ptr_, copy_size, MemcpyKind::MemcpyCUDA2CPU);
    } else {
        allocator_->memcpy(ptr_, buffer->ptr_, copy_size, MemcpyKind::MemcpyCUDA2CUDA);
    }
}

void* Buffer::ptr() {
    return ptr_;
}

const void* Buffer::ptr() const {
    return ptr_;
}

size_t Buffer::byte_size() const {
    return byte_size_;
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
    return allocator_;
}

DeviceType Buffer::device_type() const {
    return device_type_;
}

void Buffer::set_device_type(DeviceType device_type) {
    device_type_ = device_type;
}

std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
    return shared_from_this(); // 不支持 const 成员函数
}

bool Buffer::is_external() const {
    return use_external_;
}
}  // namespace base