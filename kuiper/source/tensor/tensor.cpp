#include <numeric>
#include <cuda_runtime.h>
#include "tensor/tensor.h"

namespace tensor {
template<typename T, typename U>
static size_t reduce_dims(T begin, T end, U init_val) {
    if (begin >= end) {
        return 0;
    }
    size_t size = std::accumulate(begin, end, init_val, std::multiplies<size_t>());
    return size;
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc, 
               std::shared_ptr<base::DeviceAllocator> allocator, void* ptr) : data_type_(data_type)
{
    dims_.push_back(dim0);
    size_ = dim0;
    init_buffer(allocator, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc, 
               std::shared_ptr<base::DeviceAllocator> allocator, void* ptr) : data_type_(data_type)
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0 * dim1;
    init_buffer(allocator, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc, 
               std::shared_ptr<base::DeviceAllocator> allocator, void* ptr) : data_type_(data_type)
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = dim0 * dim1 * dim2;
    init_buffer(allocator, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3, bool need_alloc, 
               std::shared_ptr<base::DeviceAllocator> allocator, void* ptr) : data_type_(data_type)
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    dims_.push_back(dim3);
    size_ = dim0 * dim1 * dim2 * dim3;
    init_buffer(allocator, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc, std::shared_ptr<base::DeviceAllocator> allocator, void* ptr)
: data_type_(data_type), dims_(std::move(dims)) {
    size_ = reduce_dims(dims_.begin(), dims_.end(), 1);
    init_buffer(allocator, need_alloc, ptr);
}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> allocator, bool need_alloc, void* ptr) {
    if (!allocator) { // 如果 allocator 为空，表示该 tensor 不会对该内存/显存 ptr 进行管理
        CHECK_EQ(need_alloc, false);
        if (!ptr) { // 当 ptr 为空时，不需要构造 buffer，此时应该通过外部 buffer 赋值(assign)进来
            return;
        }
        buffer_ = std::make_shared<base::Buffer>(byte_size(), nullptr, ptr, true);
    } else if (need_alloc) {
        allocate(allocator, false); // 如果存在合法的 buffer，则不需要 realloc
    } else {
        if (ptr) { // allocator 需要管理 ptr，因此析构函数会自动释放内存/显存
            buffer_ = std::make_shared<base::Buffer>(byte_size(), allocator, ptr, false);
        } else {
            allocate(allocator, true);
        }
    }
}

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
    // 排除 分配器为空
    if (!allocator) {
        LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!" << std::endl;
        return false;
    }
    
    // 排除 当前 Buffer 字节数为 0
    size_t byte_size_ = byte_size();
    if (!byte_size_) {
        LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!" << std::endl;
        return false;
    }
    
    // 当前 Buffer 存在，且 Tensor 字节数 <= Buffer 字节数 (合理)
    if (buffer_ && byte_size_ <= buffer_->byte_size()) {
        if (!need_realloc) { // 不需要重新分配
            return true;
        }
    }
    
    // 否则，使用 allocator 重新分配 Buffer
    buffer_ = std::make_shared<base::Buffer>(byte_size_, allocator, nullptr);

    // 判断内存是否分配成功
    if (!buffer_->ptr()) {
        LOG(ERROR) << "The memory allocated is a null pointer!" << std::endl;
        return false;
    }
    return true;
}

bool Tensor::assgin(std::shared_ptr<base::Buffer> buffer) {
    // 排除源 Buffer 为空
    if (!buffer) {
        LOG(ERROR) << "The buffer parameter in the assign function is null pointer!" << std::endl;
        return false;
    }
    if (buffer_) {
        // 目标 Buffer 与源 Buffer 设备类型不同
        if (buffer_->device_type() != buffer->device_type()) {
            LOG(ERROR) << "The device type of the new buffer is different from the original one." << std::endl;
        }
    }
    // 确保源 Buffer 字节数 >= 目标 Buffer 字节数
    if (buffer->byte_size() < byte_size()) {
        LOG(ERROR) << "The size of buffer is too small for the tensor!" << std::endl;
        return false;
    }
    buffer_ = buffer;
    return true;
}

void Tensor::to_cpu() {
    CHECK_NE(buffer_, nullptr);
    const base::DeviceType device_type_ = device_type();
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        size_t byte_size_ = byte_size();
        auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
        auto buffer_cpu = std::make_shared<base::Buffer>(byte_size_, allocator_cpu);
        allocator_cpu->memcpy(buffer_cpu->ptr(), buffer_->ptr(), byte_size_, base::MemcpyKind::MemcpyCUDA2CPU);
        buffer_ = buffer_cpu;
    } else if (device_type_ == base::DeviceType::DeviceCPU) {
        LOG(INFO) << "The device type of the tensor is already cpu." << std::endl;
    } else {
        LOG(ERROR) << "The device type of the tensor is unknown." << std::endl;
    }
}

void Tensor::to_cuda(cudaStream_t stream) {
    CHECK_NE(buffer_, nullptr);
    const base::DeviceType device_type_ = device_type();
    if (device_type_ == base::DeviceType::DeviceCPU) {
        size_t byte_size_ = byte_size();
        auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
        auto buffer_cu = std::make_shared<base::Buffer>(byte_size_, allocator_cu);
        allocator_cu->memcpy(buffer_cu->ptr(), buffer_->ptr(), byte_size_, base::MemcpyKind::MemcpyCPU2CUDA, stream);
        buffer_ = buffer_cu;
    } else if (device_type_ == base::DeviceType::DeviceCUDA) {
        LOG(INFO) << "The device type of the tensor is already cuda." << std::endl;
    } else {
        LOG(ERROR) << "The device type of the tensor is unknown." << std::endl;
    }
}

void Tensor::reshape(const std::vector<int32_t>& dims) {
    size_t size = reduce_dims(dims.begin(), dims.end(), 1);
    if (!buffer_) {
        dims_ = dims;
        size_ = size;
        return;
    }
    // 当前 tensor 的元素总数不够，需要扩容
    if (size_ < size) {
        auto buffer = std::make_shared<base::Buffer>(size * base::data_type_size(data_type_), buffer_->allocator());
        buffer->copy_from(buffer_.get());
        buffer_ = buffer;
    }
    dims_ = dims;
    size_ = size;
}

int32_t Tensor::dims_size() const {
    return static_cast<int32_t>(dims_.size());
}

int32_t Tensor::get_dim(int32_t index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, dims_.size());
    return dims_.at(index);
}

const std::vector<int32_t>& Tensor::dims() const {
    return dims_;
}

std::vector<size_t> Tensor::strides() const {
    int32_t dims_size = dims_.size();
    if (!dims_size) {
        return std::vector<size_t>();
    }
    std::vector<size_t> strides(dims_size, 1);
    for (int32_t i = dims_size - 1; i >= 1; i--) {
        strides[i - 1] = strides[i] * dims_[i];
    }
    return strides;
}

Tensor Tensor::clone() const {
    Tensor t = *this;
    size_t byte_size_ = byte_size();
    auto allocator = buffer_->allocator();
    t.buffer_ = std::make_shared<base::Buffer>(byte_size_, allocator);
    t.buffer_->copy_from(buffer_.get());
    return t;
}

void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
    size_ = reduce_dims(dims.begin(), dims.end(), 1);
    dims_ = dims;
    buffer_ = nullptr;
    data_type_ = data_type;
}

bool Tensor::is_empty() const {
    return !size_ || !buffer_ || !buffer_->ptr();
}

size_t Tensor::size() const {
    return size_;
}

size_t Tensor::byte_size() const {
    return size_ * base::data_type_size(data_type_);
}

base::DataType Tensor::data_type() const {
    return data_type_;
}

base::DeviceType Tensor::device_type() const {
    if (!buffer_) {
        return base::DeviceType::DeviceUnknown;
    }
    return buffer_->device_type();
}

void Tensor::set_device_type(base::DeviceType device_type) const {
    if (!buffer_ || buffer_->device_type() == device_type) {
        return;
    }
    buffer_->set_device_type(device_type);
}

std::shared_ptr<base::Buffer> Tensor::get_buffer() const {
    return buffer_;
}

}  // namespace tensor