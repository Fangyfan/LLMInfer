#include <unistd.h>
#include <sys/mman.h>
#include "model/raw_model_data.h"

namespace model {
RawModelData::~RawModelData() {
    // 清理内存映射 mmap
    if (data != nullptr && data != MAP_FAILED) {
        munmap(data, file_size);  // 解除内存映射
        data = nullptr;           // 置空避免悬空指针
    }
    // 关闭文件
    if (fd != -1) {
        close(fd);  // 关闭文件描述符
        fd = -1;    // 标记为已关闭
    }
}

const void* RawModelDataFp32::weight_ptr(size_t offset) const { // offset: 以 fp32 元素为单位的索引
    return reinterpret_cast<float*>(weight_data) + offset;
}

const void* RawModelDataInt8::weight_ptr(size_t offset) const { // offset: 以 int8 元素为单位的索引
    return reinterpret_cast<int8_t*>(weight_data) + offset;
}
}  // namespace model