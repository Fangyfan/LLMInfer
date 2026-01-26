#include "model/model.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string tokenizer_path, std::string model_path, bool is_quant_model)
: tokenizer_type_(tokenizer_type), model_type_(model_type), tokenizer_path_(std::move(tokenizer_path)), model_path_(std::move(model_path)), is_quant_model_(is_quant_model) {}

base::ModelType Model::model_type() const {
    return model_type_;
}

const std::string& Model::tokenizer_path() const {
    return tokenizer_path_;
}

const std::string& Model::model_path() const {
    return model_path_;
}

// 管理一个 map 容器 buffers_，用于存储不同类型的张量（Tensor）
base::Status Model::insert_buffer(ModelBufferType model_buffer_type, const tensor::Tensor& model_buffer) {
    if (buffers_.count(model_buffer_type)) {
        return base::error::key_has_exits("The inserting buffer " + std::to_string(int(model_buffer_type)) + " has exited in the model buffers");
    }
    if (model_buffer.is_empty()) {
        return base::error::invalid_argument("The inserting buffer " + std::to_string(int(model_buffer_type)) + " is empty.");
    }
    buffers_.insert({model_buffer_type, model_buffer});
    return base::error::success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType model_buffer_type) {
    CHECK(buffers_.count(model_buffer_type)) << "get buffer miss: " << int(model_buffer_type) << std::endl;
    return buffers_.at(model_buffer_type);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType model_buffer_type) const {
    CHECK(buffers_.count(model_buffer_type)) << "get buffer miss: " << int(model_buffer_type) << std::endl;
    return buffers_.at(model_buffer_type);
}

bool Model::is_sentence_end(int32_t token_id) const {
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->is_sentence_end(token_id);
}

std::string Model::decode(int32_t token_id) const {
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->decode(token_id);
}

std::string Model::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->decode(token_ids);
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->encode(sentence);
}

// 根据配置的模型类型，实例化对应的分词器
base::Status Model::create_encode_layer() {
    // 如果是 EncodeSpe，创建 SpeEncodeLayer (通常指 SentencePiece，用于 Llama 等)
    if (tokenizer_type_ == base::TokenizerType::EncodeSpe) {
        encode_layer_ = std::make_unique<op::SpeEncodeLayer>(tokenizer_path_, true, false);
    } else {
        return base::error::invalid_argument("Unknown tokenizer type in model.cpp::create_encode_layer function.");
    }
    if (!encode_layer_) {
        return base::error::internal_error("Create the encode layer failed.");
    }
    config_->vocab_size = encode_layer_->vocab_size();
    if (config_->vocab_size <= 0) {
        return base::error::internal_error("The vocab size param read error from the model file!");
    }
    return base::error::success();
}

// KV Cache 管理: 获取特定层、特定 Token 位置的 KV Cache 内存切片
// KV Cache 通常是一个巨大的预分配显存/内存块
// 这允许模型在推理时直接向特定内存位置写入当前的 K 和 V 向量
std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_id, int32_t token_pos) const {
    // 1. 指针算术：layer_offset 跳过前面的层，cache_offset 跳过当前层前面的 Token
    int32_t layer_offset = layer_id * config_->max_seq_len * config_->kv_dim;
    int32_t cache_offset = layer_offset + token_pos * config_->kv_dim;

    // 2. 切片：从总的 KeyCache 和 ValueCache buffer 中，根据计算出的 offset 拿到指针，构建一个新的 Tensor 对象返回
    const tensor::Tensor& key_cache = get_buffer(model::ModelBufferType::KeyCache);
    const tensor::Tensor& value_cache = get_buffer(model::ModelBufferType::ValueCache);

    float* key_ptr = const_cast<float*>(key_cache.ptr<float>(cache_offset));
    float* value_ptr = const_cast<float*>(value_cache.ptr<float>(cache_offset));
    
    tensor::Tensor key(base::DataType::DataTypeFp32, config_->kv_dim, false, nullptr, key_ptr);
    tensor::Tensor value(base::DataType::DataTypeFp32, config_->kv_dim, false, nullptr, value_ptr);
    key.set_device_type(device_type_);
    value.set_device_type(device_type_);
    return std::make_pair(key, value);
}

// 输入数据准备: 准备当前步骤的输入 Tensor (即当前 token 的输入 Embedding)
tensor::Tensor Model::get_embedding(const tensor::Tensor& token_pos, const op::EmbeddingResult& embedding_result, bool is_prompt) const {
    // 1. 从 embedding_result 中获取嵌入向量
    const tensor::Tensor& token_embeddings = embedding_result.token_embeddings;
    // 2. Prompt 阶段 vs 生成阶段: 
    // 如果是 is_prompt（预填充阶段），根据位置 pos 获取对应的嵌入
    // 如果是生成阶段，则 index 为 0，因为此时 embeddings 中只有一个 embedding
    int32_t index = 0;
    if (is_prompt) {
        index = token_pos.index<int32_t>(0);
    }
    // 3. 零拷贝构造：创建一个 base::Buffer，让它直接指向 embedding_output 的内存地址（避免数据拷贝），然后封装成 Tensor 返回
    float* ptr = const_cast<float*>(token_embeddings.ptr<float>(index * config_->dim));
    tensor::Tensor token_embedding(base::DataType::DataTypeFp32, config_->dim, false, nullptr, ptr);
    token_embedding.set_device_type(device_type_);
    return token_embedding;
}

// 模型权重加载: 读取模型权重文件，并将其映射到内存中
// 这是代码中最底层、也是涉及操作系统特性的部分
base::Status Model::read_model_file() {
    if (model_path_.empty()) {
        return base::error::path_not_valid("Failed to open the weight file, the model path is empty!");
    }
    
    // 1. 文件打开：使用 open 和 fopen 打开模型文件
    int32_t fd = open(model_path_.data(), O_RDONLY);
    if (fd == -1) {
        return base::error::path_not_valid("Failed to open the weight file " + model_path_ + ", may be the path does not exist!");
    }

    FILE* file = fopen(model_path_.data(), "rb");
    if (!file) {
        return base::error::path_not_valid("Failed to open the weight file " + model_path_ + ", may be the path does not exist!");
    }

    // 2. 读取头信息：先 fread 读取 ModelConfig（配置头），如果是量化模型（is_quant_model_），还会读取 group_size
    auto config = ModelConfig();
    if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
        return base::error::model_parse_error("Failed to retrieve the configuration information from the model file.");
    }
    if (is_quant_model_) {
        if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
            return base::error::model_parse_error("Failed to retrieve the group size information from the model file.");
        }
    }

    // 3. 生成配置：调用 generate_model_info 处理刚读到的配置
    base::Status status = generate_model_info(config);
    if (!status) {
        return status;
    }

    // 4. mmap (内存映射)：这是高性能推理的关键
    // 代码并没有使用 fread 把几 GB 的权重全部读入 RAM，而是使用了 mmap
    // mmap 将文件直接映射到进程的虚拟地址空间，操作系统会根据需要（缺页中断）按需加载数据，这使得加载速度极快且节省内存
    struct stat st;
    if (fstat(fd, &st) == -1) {
        close(fd);
        return base::error::model_parse_error("Failed to retrieve the file size information from the model file.");
    }
    size_t file_size = st.st_size;
    void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED || data == nullptr) {
        return base::error::model_parse_error("Failed to map the weight file " + model_path_ + " into memory.");
    }
    
    // 5. 指针定位：计算权重数据在内存中的起始位置：weight_data = data (起始地址) + sizeof(ModelConfig) (跳过头信息)
    if (!is_quant_model_) {
        raw_model_data_ = std::make_unique<RawModelDataFp32>();
    } else {
        raw_model_data_ = std::make_unique<RawModelDataInt8>();
    }
    raw_model_data_->data = data;
    raw_model_data_->fd = fd;
    raw_model_data_->file_size = file_size;
    raw_model_data_->weight_data = static_cast<char*>(data) + sizeof(ModelConfig) + (is_quant_model_ ? sizeof(group_size_) : 0);
    return base::error::success();
}

// 这是模型加载的主入口函数
base::Status Model::create_model() {
    // 1. 创建配置对象 TransformerConfig
    config_ = std::make_unique<TransformerConfig>();

    // 2. 调用 create_encode_layer() 初始化分词器
    base::Status status = create_encode_layer();
    if (!status) {
        LOG(ERROR) << "Create the encode layer failed!" << std::endl;
        return status;
    }
    
    // 3. 调用 read_model_file() 将模型权重加载到内存 (使用 mmap 内存映射)
    status = read_model_file();
    if (!status) {
        LOG(ERROR) << "Handle model file " << model_path_ << " failed!" << std::endl;
        return status;
    }
    
    // 4. 调用 create_layers() 构建具体的网络层
    status = create_layers();
    if (!status) {
        LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!" << std::endl;
        return status;
    }
    return base::error::success();
}

// 解析从文件头读取的 ModelConfig 结构体，将其转换为内部使用的配置格式
base::Status Model::generate_model_info(const ModelConfig& config) const {
    // 1. 参数拷贝：将 dim (维度), layer_num (层数), head_num (头数) 等从文件配置拷贝到类成员
    config_->dim = config.dim;
    config_->head_num = config.head_num;
    config_->hidden_dim = config.hidden_dim;
    config_->kv_head_num = config.kv_head_num;
    config_->layer_num = config.layer_num;
    config_->max_seq_len = config.max_seq_len;
    
    // 2. 计算 KV 参数 (kv_dim, kv_mul, head_dim)：这里包含了 GQA (Grouped Query Attention) 的逻辑
    CHECK(config_->dim % config_->head_num == 0);
    config_->head_dim = config_->dim / config_->head_num;
    config_->kv_dim = config_->head_dim * config_->kv_head_num;
    CHECK(config_->head_dim % 2 == 0);

    CHECK(config_->head_num % config_->kv_head_num == 0);
    config_->kv_mul = config_->head_num / config_->kv_head_num;

    // 3. 根据 vocab_size 的正负性，判断 embedding (vocab_size, dim) 层和 output (dim, vocab_size) 层是否共享权重
    config_->is_shared_weight = config.vocab_size > 0;

    // 4. 词表大小校正：针对 Llama / Qwen 模型处理词表大小可能为负数或不匹配的情况
    config_->vocab_size = std::abs(config.vocab_size);

    // LOG(INFO) << "dim = " << config_->dim << std::endl;
    // LOG(INFO) << "head_dim = " << config_->head_dim << std::endl;
    // LOG(INFO) << "head_num = " << config_->head_num << std::endl;
    // LOG(INFO) << "hidden_dim = " << config_->hidden_dim << std::endl;
    // LOG(INFO) << "is_shared_weight = " << config_->is_shared_weight << std::endl;
    // LOG(INFO) << "kv_dim = " << config_->kv_dim << std::endl;
    // LOG(INFO) << "kv_head_num = " << config_->kv_head_num << std::endl;
    // LOG(INFO) << "kv_mul = " << config_->kv_mul << std::endl;
    // LOG(INFO) << "layer_num = " << config_->layer_num << std::endl;
    // LOG(INFO) << "max_seq_len = " << config_->max_seq_len << std::endl;
    // LOG(INFO) << "vocab_size = " << config_->vocab_size << std::endl;
    return base::error::success();
}
}  // namespace model