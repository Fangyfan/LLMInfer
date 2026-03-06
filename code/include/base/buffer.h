#ifndef KUIPER_INCLUDE_BASE_BUFFER_H
#define KUIPER_INCLUDE_BASE_BUFFER_H

#include "base/alloc.h"

namespace base {
/*
情况1: 为 Buffer 分配一块所属的内存，使用内存分配器分配得到的内存指针是 ptr_，并且将 use_external 置为 false
表示在 Buffer 对象析构的时候也要将相关联的资源指针进行释放
如果表示 Buffer 拥有当前的内存/显存，那么我们就在构造函数中指定 use_external 为 false（表示当前 Buffer 拥有该对象）
我们使用 Buffer(32, allocator)，随后构造函数就会自动调用 allocate（上文说过的 DeviceAllocator）方法完成资源的申请

情况2: Buffer 不申请资源，传入的指针指向并不属于它，因为不具有所属权，所以在退出的时候不需要对它释放
我们只需要使用 Buffer buffer(32, nullptr, ptr, true) 实例化，这表示我们将 ptr 指针赋值给 Buffer
但是 use_external = true 变量表示 buffer 实例不需要负责去释放 ptr
*/
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
public:
    explicit Buffer() = default;
    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr, void* ptr = nullptr, bool use_external = false);
    virtual ~Buffer();
    bool allocate();
    void copy_from(const Buffer& buffer) const;
    void copy_from(const Buffer* buffer) const;
    void* ptr();
    const void* ptr() const;
    size_t byte_size() const;
    std::shared_ptr<DeviceAllocator> allocator() const;
    DeviceType device_type() const;
    void set_device_type(DeviceType device_type);
    std::shared_ptr<Buffer> get_shared_from_this();
    bool is_external() const;
private:
    // 这块内存的大小，以字节数作为单位
    size_t byte_size_ = 0;
    // Buffer 对应设备类型的内存分配器，负责资源的释放、申请以及拷贝等，既可以是 cpu allocator 也可以是 cuda allocator
    std::shared_ptr<DeviceAllocator> allocator_;
    // 这块内存的地址
    // 主要有两种来源，一种是外部直接赋值得到的，Buffer 不需要对它进行管理，和它的关系是借用，不负责它的生命周期管理，这种情况下 use_external = true。
    void* ptr_ = nullptr;
    // 是否拥有这块数据所有权（即是否使用外部数据，不对其释放）
    bool use_external_ = false;
    // Buffer 中内存资源所属的设备类型
    DeviceType device_type_ = DeviceType::DeviceUnknown;
};
}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_BUFFER_H