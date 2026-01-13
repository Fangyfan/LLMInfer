#ifndef KUIPER_INCLUDE_MODEL_RAW_MODEL_DATA_H
#define KUIPER_INCLUDE_MODEL_RAW_MODEL_DATA_H

#include <cstdint>
#include <cstddef>

namespace model {
// 非量化模型使用 RawModelDataFp32，量化模型使用 RawModelDataInt8
struct RawModelData {
    ~RawModelData();
    int32_t fd = -1; // 文件描述符，用于内存映射系统调用
    size_t file_size = 0; // 模型文件的总字节数
    void* data = nullptr; // 整个文件映射到内存的起始地址
    void* weight_data = nullptr; // 权重数据在内存中的起始地址

    // 第 offset 个权重数据 (fp32 / int8) 在内存中地址
    virtual const void* weight_ptr(size_t offset) const = 0;
};

struct RawModelDataFp32 : RawModelData {
    const void* weight_ptr(size_t offset) const override;
};

struct RawModelDataInt8 : RawModelData {
    const void* weight_ptr(size_t offset) const override;
};
}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_RAW_MODEL_DATA_H