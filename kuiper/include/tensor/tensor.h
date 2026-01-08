#ifndef KUIPER_INCLUDE_TENSOR_TENSOR_H
#define KUIPER_INCLUDE_TENSOR_TENSOR_H

#include <cuda_runtime_api.h>
#include "base/buffer.h"

namespace tensor {
class Tensor {
public:
    explicit Tensor() = default;
    
    // need_alloc 表示是否需要用内存分配器 allocator 来分配内存/显存
    explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false, 
                    std::shared_ptr<base::DeviceAllocator> allocator = nullptr, void* ptr = nullptr);
    
    // need_alloc 表示是否需要用内存分配器 allocator 来分配内存/显存
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false, 
                    std::shared_ptr<base::DeviceAllocator> allocator = nullptr, void* ptr = nullptr);
    
    // need_alloc 表示是否需要用内存分配器 allocator 来分配内存/显存
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc = false, 
                    std::shared_ptr<base::DeviceAllocator> allocator = nullptr, void* ptr = nullptr);
    
    // need_alloc 表示是否需要用内存分配器 allocator 来分配内存/显存
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3, bool need_alloc = false, 
                    std::shared_ptr<base::DeviceAllocator> allocator = nullptr, void* ptr = nullptr);
    
    // need_alloc 表示是否需要用内存分配器 allocator 来分配内存/显存
    explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false, 
                    std::shared_ptr<base::DeviceAllocator> allocator = nullptr, void* ptr = nullptr);
    
    void init_buffer(std::shared_ptr<base::DeviceAllocator> allocator, bool need_alloc, void* ptr);
    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc); // 分配内存(Buffer)
    bool assgin(std::shared_ptr<base::Buffer> buffer); // 赋值 Buffer
    
    void to_cpu(); // 将数据从 CUDA 转移到 CPU
    void to_cuda(cudaStream_t stream = nullptr); // 将数据从 CPU 转移到 CUDA
    
    void reshape(const std::vector<int32_t>& dims); // 改变形状（不改变数据）
    int32_t dims_size() const; // 获取维度的数量
    int32_t get_dim(int32_t index) const; // 获取某个维度大小
    const std::vector<int32_t>& dims() const; // 获取所有维度
    std::vector<size_t> strides() const; // 计算步长（内存布局）
    
    Tensor clone() const; // 深拷贝
    void reset(base::DataType data_type, const std::vector<int32_t>& dims); // 重置张量，Buffer -> nullptr
    
    bool is_empty() const; // 是否为空
    size_t size() const; // 元素总数
    size_t byte_size() const;  // 字节大小
    base::DataType data_type() const; // 数据类型
    base::DeviceType device_type() const; // 设备类型（CPU/CUDA）
    void set_device_type(base::DeviceType device_type) const; // 指定设备类型
    std::shared_ptr<base::Buffer> get_buffer() const; // 获取 Buffer 指针
    
    template<typename T>
    T* ptr(); // 获取张量起始地址
    
    template<typename T>
    const T* ptr() const; // 获取张量起始地址
    
    template<typename T>
    T* ptr(int64_t index); // 获取某个偏移量的元素地址
    
    template<typename T>
    const T* ptr(int64_t index) const; // 获取某个偏移量的元素地址
    
    template<typename T>
    T& index(int64_t offset); // 获取某个偏移量的元素
    
    template<typename T>
    const T& index(int64_t offset) const; // 获取某个偏移量的元素

private:
    // 张量中数据的个数，比如张量后期要存储 3 个数据，分别为 {1，2，3}，那么 size 的大小就等于 3
    size_t size_ = 0;
    
    // 张量的维度，比如有一个二维张量，且维度分别是 {2, 3}，那么 dim_ 记录的值就是 {2, 3}
    std::vector<int32_t> dims_;
    
    // Buffer 类用来管理用分配器申请到的内存资源
    // buffer->allocator 会根据 buffer 中的设备类型去申请对应设备上的内存/显存资源，并在 buffer 实例析构的时候自动释放相关的内存/显存资源
    std::shared_ptr<base::Buffer> buffer_;

    base::DataType data_type_ = base::DataType::DataTypeUnknown;
};

template<typename T>
T* Tensor::ptr() {
    if (!buffer_) {
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_->ptr());
}

template<typename T>
const T* Tensor::ptr() const {
    if (!buffer_) {
        return nullptr;
    }
    return reinterpret_cast<const T*>(buffer_->ptr());
}

template<typename T>
T* Tensor::ptr(int64_t index) {
    CHECK(buffer_ && buffer_->ptr()) << "The data area buffer of this tensor is empty or it points to a null pointer." << std::endl;
    return reinterpret_cast<T*>(buffer_->ptr()) + index;
}

template<typename T>
const T* Tensor::ptr(int64_t index) const {
    CHECK(buffer_ && buffer_->ptr()) << "The data area buffer of this tensor is empty or it points to a null pointer." << std::endl;
    return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

template<typename T>
T& Tensor::index(int64_t offset) {
    CHECK_GE(offset, 0);
    CHECK_LT(offset, size_);
    T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
}

template<typename T>
const T& Tensor::index(int64_t offset) const {
    CHECK_GE(offset, 0);
    CHECK_LT(offset, size_);
    const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
}

}  // namespace tensor

#endif  // KUIPER_INCLUDE_TENSOR_TENSOR_H