#ifndef KUIPER_INCLUDE_BASE_BASE_H
#define KUIPER_INCLUDE_BASE_BASE_H

#include <glog/logging.h>
#include <cstdint>
#include <string>

// UNUSED 宏：显式标记未使用的参数，消除编译器警告
#define UNUSED(expr) do { (void)(expr); } while (0)

namespace model {
enum class ModelBufferType : uint8_t {
    InputTokens = 0,       // 输入 tokens
    InputEmbeddings = 1,   // 输入词嵌入
    OutputRMSNorm = 2,     // RMS 归一化输出
    KeyCache = 3,          // 注意力机制的 Key 缓存
    ValueCache = 4,        // 注意力机制的 Value 缓存
    Query = 5,             // 查询向量
    InputPos = 6,          // 输入位置信息
    ScoreStorage = 7,      // 注意力分数存储
    OutputMHA = 8,         // 多头注意力输出
    AttentionOutput = 9,   // 注意力输出
    W1Output = 10,         // 前馈网络第一层输出
    W2Output = 11,         // 前馈网络第二层输出
    W3Output = 12,         // 前馈网络第三层输出
    FFNRMSNorm = 13,       // FFN 的 RMS 归一化
    ForwardOutput = 14,    // 前向传播输出
    ForwardOutputCPU = 15, // CPU 上的前向传播输出
    SinCache = 16,         // 正弦位置编码缓存
    CosCache = 17,         // 余弦位置编码缓存
};
}  // namespace model

namespace base {
enum class DeviceType : uint8_t {
    DeviceUnknown = 0,
    DeviceCPU = 1,
    DeviceCUDA = 2,
};

enum class DataType : uint8_t {
    DataTypeUnknown = 0,
    DataTypeFp32 = 1,    // 单精度浮点
    DataTypeInt8 = 2,    // 8 位整数（量化）
    DataTypeInt32 = 3,   // 32 位整数
};

enum class ModelType : uint8_t {
    ModelTypeUnown = 0,
    ModelTypeLLama2 = 1,
};

inline size_t data_type_size(DataType data_type) {
    if (data_type == DataType::DataTypeFp32) {
        return sizeof(float);
    } else if (data_type == DataType::DataTypeInt8) {
        return sizeof(int8_t);
    } else if (data_type == DataType::DataTypeInt32) {
        return sizeof(int32_t);
    } else {
        LOG(FATAL) << "Unknown data type size for " << int(data_type) << std::endl;
        return 0;
    }
}

// 禁止拷贝，防止对象被意外拷贝，作为基类继承
class NoCopyable {
protected:
    NoCopyable() = default;                             // 默认构造函数
    ~NoCopyable() = default;                            // 默认析构函数
    NoCopyable(const NoCopyable&) = delete;             // 禁止拷贝构造
    NoCopyable& operator=(const NoCopyable&) = delete;  // 禁止拷贝赋值
};

// 统一的错误码系统
enum class StatusCode : uint8_t {
    Success = 0,               // 成功
    FunctionUnImplement = 1,   // 功能未实现
    PathNotValid = 2,          // 路径无效
    ModelParseError = 3,       // 模型解析错误
    InternalError = 5,         // 内部错误
    KeyValueHasExist = 6,      // 键值已存在
    InvalidArgument = 7,       // 无效参数
};

enum class TokenizerType : int8_t {
    EncodeUnknown = -1,
    EncodeSpe = 0,      // SentencePiece 分词器
    EncodeBpe = 1,      // BPE 分词器
};

class Status {
public:
    Status(StatusCode code = StatusCode::Success, std::string err_msg = "");
    Status(const Status& other) = default;
    Status& operator=(const Status& other) = default;
    Status& operator=(StatusCode code);
    bool operator==(StatusCode code) const;
    bool operator!=(StatusCode code) const;
    operator int() const;
    operator bool() const;
    StatusCode get_err_code() const;
    const std::string& get_err_msg() const;
    void set_err_msg(const std::string& err_msg);
private:
    StatusCode code_ = StatusCode::Success;
    std::string err_msg_;
};

namespace error {
// 1. 执行传入的函数 call，并将返回的 Status 赋值给常量引用
// 2. 检查 Status 是否为失败状态
// 3. 定义一个 512 字节的字符数组，用于存储错误信息
// 4. 格式化错误信息到 buf 中（关键：包含文件、行号、错误码、错误描述）
#define STATUS_CHECK(call)                                                                  \
    do {                                                                                    \
        const base::Status& status = call;                                                  \
        if (!status) {                                                                      \
            const size_t buf_size = 512;                                                    \
            char buf[buf_size];                                                             \
            snprintf(buf, buf_size,                                                         \
                     "Infer error\n File: %s Line: %d\n Error code: %d\n Error msg: %s\n",  \
                     __FILE__, __LINE__, int(status), status.get_err_msg().c_str());        \
            LOG(FATAL) << buf;                                                              \
        }                                                                                   \
    } while(0)

Status success(const std::string& err_msg = "");
Status function_not_implement(const std::string& err_msg = "");
Status path_not_valid(const std::string& err_msg = "");
Status model_parse_error(const std::string& err_msg = "");
Status internal_error(const std::string& err_msg = "");
Status key_has_exits(const std::string& err_msg = "");
Status invalid_argument(const std::string& err_msg = "");
}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& status);
}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_BASE_H