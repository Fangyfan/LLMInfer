#ifdef QWEN3_SUPPORT
#include "model/qwen3.h"
#include "op/add.h"
#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"
#include "op/rope.h"
#include "op/swiglu.h"
#include "../op/kernel/kernel_interface.h"

namespace model {
void Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> cuda_config) {
    CHECK_NE(cuda_config, nullptr);
    embedding_layer_->set_cuda_config(cuda_config);
    swiglu_layer_->set_cuda_config(cuda_config);
    rope_layer_->set_cuda_config(cuda_config);
    add_layer_->set_cuda_config(cuda_config);
    mha_layer_->set_cuda_config(cuda_config);
    cls_layer_->set_cuda_config(cuda_config);

    embedding_layer_->to_cuda();
    swiglu_layer_->to_cuda();
    rope_layer_->to_cuda();
    add_layer_->to_cuda();
    mha_layer_->to_cuda();
    cls_layer_->to_cuda();

    for (auto& rmsnorm_layer : rmsnorm_layers_) {
        rmsnorm_layer->set_cuda_config(cuda_config);
        rmsnorm_layer->to_cuda();
    }
    for (auto& wq_layer : wq_layers_) {
        wq_layer->set_cuda_config(cuda_config);
        wq_layer->to_cuda();
    }
    for (auto& wk_layer : wk_layers_) {
        wk_layer->set_cuda_config(cuda_config);
        wk_layer->to_cuda();
    }
    for (auto& wv_layer : wv_layers_) {
        wv_layer->set_cuda_config(cuda_config);
        wv_layer->to_cuda();
    }
    for (auto& wo_layer : wo_layers_) {
        wo_layer->set_cuda_config(cuda_config);
        wo_layer->to_cuda();
    }
    for (auto& w1_layer : w1_layers_) {
        w1_layer->set_cuda_config(cuda_config);
        w1_layer->to_cuda();
    }
    for (auto& w2_layer : w2_layers_) {
        w2_layer->set_cuda_config(cuda_config);
        w2_layer->to_cuda();
    }
    for (auto& w3_layer : w3_layers_) {
        w3_layer->set_cuda_config(cuda_config);
        w3_layer->to_cuda();
    }
}

Qwen3Model::Qwen3Model(base::TokenizerType tokenizer_type, std::string tokenizer_path, std::string model_path, bool is_quant_model)
: Model(tokenizer_type, base::ModelType::ModelTypeQwen3, std::move(tokenizer_path), std::move(model_path), is_quant_model) {}

base::Status Qwen3Model::init(base::DeviceType device_type) {
    // 1. 设备检查与环境搭建: token path 检查，CPU 量化检查, CUDA 初始化
    if (tokenizer_path_.empty()) {
        return base::error::path_not_valid(tokenizer_path_);
    }
    if (is_quant_model_) {
        return base::error::invalid_argument("Qwen3 do not support int8 quant model.");
    }
    
    device_type_ = device_type;
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        if (cudaSetDevice(3) != cudaSuccess) {
            return base::error::internal_error("The cuda set device id failed.");
        }
        cuda_config_ = std::make_shared<kernel::CudaConfig>();
        cuda_config_->create();
    }
    // 2. 模型加载与缓冲区分配(内存/显存)
    base::Status status = create_model();
    if (!status) {
        return status;
    }
    allocate_model_buffers();
    
    // 3. 预计算 RoPE 旋转位置编码 Sin/Cos Cache
    const tensor::Tensor& sin_cache = get_buffer(ModelBufferType::SinCache);
    const tensor::Tensor& cos_cache = get_buffer(ModelBufferType::CosCache);
    kernel::get_sin_cos_cache_kernel(device_type_)(sin_cache, cos_cache, config_->head_dim, config_->max_seq_len, 
                                                   cuda_config_ ? cuda_config_->stream : nullptr);
    // 4. 采样器初始化
    sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
    return base::error::success();
}

base::Status Qwen3Model::predict(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos, bool is_prompt, int32_t& next_token_id) const {
    base::Status status = forward(token_embedding, token_pos);
    if (!status) {
        return status;
    }
    next_token_id = post_process(is_prompt);
    return base::error::success();
}

base::Status Qwen3Model::forward(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos) const {
    if (token_embedding.is_empty()) {
        return base::error::invalid_argument("The token_embedding in Qwen3Model::forward is empty.");
    }
    if (token_pos.is_empty()) {
        return base::error::invalid_argument("The token_pos in Qwen3Model::forward is empty.");
    }
    if (token_embedding.get_dim(0) != config_->hidden_dim) {
        return base::error::invalid_argument("The token_embedding in Qwen3Model::forward has a wrong dim");
    }
    if (token_pos.get_dim(0) != 1) {
        return base::error::invalid_argument("The token_pos in Qwen3Model::forward has a wrong dim");
    }
    // 遍历所有 Transformer Block
    for (int32_t layer_id = 0; layer_id < config_->layer_num; ++layer_id) {
        // 1. 对输入 RMSNorm 归一化
        attention_rmsnorm(layer_id, token_embedding);
        // 2. 计算 QKV -> 缓存 KV -> RoPE 旋转位置编码
        attention_qkv_rope(layer_id, token_pos);
        // 3. 多头注意力
        attention_mha(layer_id, token_pos);
        // 4. 前馈网络 (FFN)
        feed_forward(layer_id, token_embedding);
    }
    // 5. 最终分类
    cls_logits(token_embedding);
    return base::error::success();
}

op::EmbeddingResult Qwen3Model::embedding(const std::vector<int32_t>& token_ids) const {
    // 1. 获取 Buffer: 从之前准备好的资源池里拿出 TokenIds 和 TokenEmbeddings
    tensor::Tensor token_ids_ = get_buffer(model::ModelBufferType::TokenIds);
    tensor::Tensor token_embeddings_ = get_buffer(model::ModelBufferType::TokenEmbeddings);
    CHECK(token_embeddings_.dims_size() == 2);

    // 2. 动态 Reshape: 虽然 Buffer 是预分配的，但预填充阶段和生成阶段的 token num 不同
    // reshape 操作通常只是修改 Tensor 的元数据（维度信息），只要新大小不超过预分配的容量，就不会触发昂贵的内存重分配
    int32_t size = static_cast<int32_t>(token_ids.size());
    if (token_ids_.size() != size) {
        token_ids_.reshape({ size });
        token_embeddings_.reshape({ size, config_->hidden_dim });
    }

    // 3. 数据填充: 把输入的 std::vector<int> 拷贝到 Buffer 中
    memcpy(token_ids_.ptr<int32_t>(), token_ids.data(), size * sizeof(int32_t));
    
    // 4. 执行查找: 本质上是查表，根据 token id 从巨大的 Embedding 矩阵中把对应的行复制出来
    tensor::Tensor token_num_(base::DataType::DataTypeInt32, size);
    STATUS_CHECK(qwen3_layers_->embedding_layer_->forward(token_ids_, token_num_, token_embeddings_));
    CHECK(token_embeddings_.dims_size() == 2);
    return op::EmbeddingResult(token_ids_, token_embeddings_, token_num_);
}

base::Status Qwen3Model::create_layers() {
    // 1. 先创建容器 qwen3_layers_
    CHECK(qwen3_layers_ == nullptr);
    qwen3_layers_ = std::make_unique<Qwen3Layers>();

    // 2. 创建无参数算子 create_nonparam_layers
    create_nonparam_layers();
    
    // 3. 根据是否量化，分别调用 create_param_layers 或 create_param_quant_layers
    if (!is_quant_model_) {
        create_param_layers();
    } else {
        return base::error::function_not_implement("");
    }
    
    // 5. 算子层数量和空指针检查
    if (!qwen3_layers_->embedding_layer_) {
        return base::error::internal_error("Create the embedding layer for the qwen3 model failed!");
    }
    if (qwen3_layers_->rmsnorm_layers_.size() != 4 * config_->layer_num + 1) {
        return base::error::internal_error("Create the rmsnorm layers for the qwen3 model failed!");
    }
    if (qwen3_layers_->wq_layers_.size() != config_->layer_num ||
        qwen3_layers_->wk_layers_.size() != config_->layer_num ||
        qwen3_layers_->wv_layers_.size() != config_->layer_num ||
        qwen3_layers_->wo_layers_.size() != config_->layer_num ||
        qwen3_layers_->w1_layers_.size() != config_->layer_num ||
        qwen3_layers_->w2_layers_.size() != config_->layer_num ||
        qwen3_layers_->w3_layers_.size() != config_->layer_num) {
        return base::error::internal_error("Create the matmul layer in the MHA and FFN layers for qwen3 model failed.");
    }
    for (int32_t i = 0; i < config_->layer_num; ++i) {
        if (!qwen3_layers_->wq_layers_[i] || !qwen3_layers_->wk_layers_[i] ||
            !qwen3_layers_->wv_layers_[i] || !qwen3_layers_->wo_layers_[i] ||
            !qwen3_layers_->w1_layers_[i] || !qwen3_layers_->w2_layers_[i] || !qwen3_layers_->w3_layers_[i]) {
            return base::error::internal_error("Create the matmul layer in the MHA and FFN layers for qwen3 model failed.");
        }
    }
    if (!qwen3_layers_->mha_layer_) {
        return base::error::internal_error("Create the mha layer for the qwen3 model failed!");
    }
    if (!qwen3_layers_->rope_layer_) {
        return base::error::internal_error("Create the rope layer for the qwen3 model failed!");
    }
    if (!qwen3_layers_->swiglu_layer_) {
        return base::error::internal_error("Create the swiglu layer for the qwen3 model failed!");
    }
    if (!qwen3_layers_->add_layer_) {
        return base::error::internal_error("Create the add layer for the qwen3 model failed!");
    }
    if (!qwen3_layers_->cls_layer_) {
        return base::error::internal_error("Create the cls logits layer for the qwen3 model failed!");
    }

    // 5. CUDA 权重搬运: 如果是 CUDA 模式，调用 qwen3_layers_->to_cuda(...) 把所有权重层移到 GPU 上
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        qwen3_layers_->to_cuda(cuda_config_);
    }
    return base::error::success();
}

void Qwen3Model::create_nonparam_layers() {
    CHECK(qwen3_layers_ != nullptr);

    int32_t dim = config_->dim;
    int32_t kv_dim = config_->kv_dim;
    int32_t kv_mul = config_->kv_mul;
    int32_t head_num = config_->head_num;
    int32_t head_dim = config_->head_dim;
    int32_t max_seq_len = config_->max_seq_len;
    int32_t immediate_dim = config_->immediate_dim;

    // 1. RoPELayer: 创建旋转位置编码算子
    // 注意参数：dim (总维度), kv_dim (KV维度，用于 GQA/MQA), head_dim (每个头的维度)
    qwen3_layers_->rope_layer_ = std::make_unique<op::RoPELayer>(device_type_, dim, kv_dim, head_dim);

    // 2. MultiHeadAttention: 创建注意力算子
    // kv_mul_: 这个参数很有意思。如果 kv_mul_ > 1，说明使用了 GQA (Grouped Query Attention)，即多个 Query 头共享一组 KV 头
    qwen3_layers_->mha_layer_ = std::make_unique<op::MultiHeadAttention>(device_type_, kv_dim, kv_mul, head_num, head_dim, max_seq_len);

    // 3. AddLayer: 向量加法算子
    // 专门用于处理残差连接（Residual Connection），即 Output = Input + F(Input)
    qwen3_layers_->add_layer_ = std::make_unique<op::AddLayer>(device_type_);

    // 4. SwiGLULayer: SwiGLU 激活算子
    // 这是 qwen3 相比原始 Transformer（使用 ReLU）的一大改进，能提供更好的非线性表达能力
    qwen3_layers_->swiglu_layer_ = std::make_unique<op::SwiGLULayer>(device_type_, immediate_dim);
}

void Qwen3Model::create_param_layers() {
    CHECK(!is_quant_model_);
    CHECK(qwen3_layers_ != nullptr);
    
    size_t offset = 0;
    int32_t dim = config_->dim;
    int32_t kv_dim = config_->kv_dim;
    int32_t head_dim = config_->head_dim;
    int32_t layer_num = config_->layer_num;
    int32_t vocab_size = config_->vocab_size;
    int32_t hidden_dim = config_->hidden_dim;
    int32_t max_seq_len = config_->max_seq_len;
    int32_t immediate_dim = config_->immediate_dim;
    
    // 1. RMSNorm (Attention, FFN, Final) : [2 * layers + 1, hidden_dim]
    for (int32_t i = 0; i < 2 * layer_num + 1; ++i) {
        auto rmsnorm = std::make_unique<op::RMSNormLaryer>(device_type_, hidden_dim);
        rmsnorm->set_weight(0, {hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->rmsnorm_layers_.push_back(std::move(rmsnorm));
        offset += hidden_dim;
    }

    // 2. Embedding : [vocab_size, hidden_dim]
    qwen3_layers_->embedding_layer_ = std::make_unique<op::EmbeddingLayer>(device_type_, hidden_dim, max_seq_len, vocab_size);
    qwen3_layers_->embedding_layer_->set_weight(0, {vocab_size, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
    offset += vocab_size * hidden_dim;

    // 3. Attention Wq : [layers, dim, hidden_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wq = std::make_unique<op::MatmulLayer>(device_type_, dim, hidden_dim);
        wq->set_weight(0, {dim, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->wq_layers_.push_back(std::move(wq));
        offset += dim * hidden_dim;
    }

    // 4. Attention q_norm : [layers, head_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto q_norm = std::make_unique<op::RMSNormLaryer>(device_type_, head_dim);
        q_norm->set_weight(0, {head_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->rmsnorm_layers_.push_back(std::move(q_norm));
        offset += head_dim;
    }

    // 5. Attention Wk : [layers, kv_dim, hidden_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wk = std::make_unique<op::MatmulLayer>(device_type_, kv_dim, hidden_dim);
        wk->set_weight(0, {kv_dim, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->wk_layers_.push_back(std::move(wk));
        offset += kv_dim * hidden_dim;
    }

    // 6. Attention k_norm : [layers, head_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto k_norm = std::make_unique<op::RMSNormLaryer>(device_type_, head_dim);
        k_norm->set_weight(0, {head_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->rmsnorm_layers_.push_back(std::move(k_norm));
        offset += head_dim;
    }

    // 7. Attention Wv : [layers, kv_dim, hidden_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wv = std::make_unique<op::MatmulLayer>(device_type_, kv_dim, hidden_dim);
        wv->set_weight(0, {kv_dim, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->wv_layers_.push_back(std::move(wv));
        offset += kv_dim * hidden_dim;
    }

    // 8. Attention Wo : [layers, hidden_dim, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wo = std::make_unique<op::MatmulLayer>(device_type_, hidden_dim, dim);
        wo->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->wo_layers_.push_back(std::move(wo));
        offset += hidden_dim * dim;
    }

    // 9. FFN W1 (Gate)	: [layers, immediate_dim, hidden_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w1 = std::make_unique<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim);
        w1->set_weight(0, {immediate_dim, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->w1_layers_.push_back(std::move(w1));
        offset += immediate_dim * hidden_dim;
    }

    // 10. FFN W2 (Down) : [layers, hidden_dim, immediate_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w2 = std::make_unique<op::MatmulLayer>(device_type_, hidden_dim, immediate_dim);
        w2->set_weight(0, {hidden_dim, immediate_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->w2_layers_.push_back(std::move(w2));
        offset += hidden_dim * immediate_dim;
    }

    // 11. FFN W3 (Up) : [layers, immediate_dim, hidden_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w3 = std::make_unique<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim);
        w3->set_weight(0, {immediate_dim, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_layers_->w3_layers_.push_back(std::move(w3));
        offset += immediate_dim * hidden_dim;
    }

    // 13. Output Head : [vocab_size, hidden_dim]
    qwen3_layers_->cls_layer_ = std::make_unique<op::MatmulLayer>(device_type_, vocab_size, hidden_dim);
    qwen3_layers_->cls_layer_->set_weight(0, {vocab_size, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
}

void Qwen3Model::create_param_quant_layers() {}

void Qwen3Model::allocate_model_buffers() {
    // 1. 分配器选择: 根据配置决定是用 CPU 内存（malloc）还是 GPU 显存（cudaMalloc）
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type_ == base::DeviceType::DeviceCPU) {
        allocator = allocator_cpu;
    } else if (device_type_ == base::DeviceType::DeviceCUDA) {
        allocator = allocator_cu;
    } else {
        LOG(FATAL) << "Unknown device type in Qwen3Model::allocate_model_buffers." << std::endl;
    }

    // 2. 核心 Buffer 分配 (预分配策略): KV Cache，RoPE 缓存 (sin_cache, cos_cache)
    // Buffer 复用 (Memory Reuse): 同一个 rms_output 张量被注册到了多个不同的 Buffer ID 上
    int32_t dim = config_->dim;
    int32_t kv_dim = config_->kv_dim;
    int32_t head_num = config_->head_num;
    int32_t head_dim = config_->head_dim;
    int32_t layer_num = config_->layer_num;
    int32_t vocab_size = config_->vocab_size;
    int32_t hidden_dim = config_->hidden_dim;
    int32_t max_seq_len = config_->max_seq_len;
    int32_t immediate_dim = config_->immediate_dim;

    tensor::Tensor token_ids(base::DataType::DataTypeInt32, 1, true, allocator_cpu);
    tensor::Tensor token_pos(base::DataType::DataTypeInt32, 1, true, allocator_cpu);
    insert_buffer(ModelBufferType::TokenIds, token_ids);
    insert_buffer(ModelBufferType::TokenPosition, token_pos);

    tensor::Tensor token_embeddings(base::DataType::DataTypeFp32, 1, hidden_dim, true, allocator);
    insert_buffer(ModelBufferType::TokenEmbeddings, token_embeddings);

    tensor::Tensor sin_cache(base::DataType::DataTypeFp32, max_seq_len, head_dim / 2, true, allocator);
    tensor::Tensor cos_cache(base::DataType::DataTypeFp32, max_seq_len, head_dim / 2, true, allocator);
    insert_buffer(ModelBufferType::SinCache, sin_cache);
    insert_buffer(ModelBufferType::CosCache, cos_cache);

    tensor::Tensor key_cache(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator);
    tensor::Tensor value_cache(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator);
    insert_buffer(ModelBufferType::KeyCache, key_cache);
    insert_buffer(ModelBufferType::ValueCache, value_cache);

    tensor::Tensor rmsnorm(base::DataType::DataTypeFp32, hidden_dim, true, allocator);
    insert_buffer(ModelBufferType::MHAPreRMSNorm, rmsnorm);
    insert_buffer(ModelBufferType::FFNPreRMSNorm, rmsnorm);
    insert_buffer(ModelBufferType::FFNW2Output, rmsnorm);

    tensor::Tensor query(base::DataType::DataTypeFp32, dim, true, allocator);
    insert_buffer(ModelBufferType::Query, query);

    tensor::Tensor score(base::DataType::DataTypeFp32, head_num, max_seq_len, true, allocator);
    insert_buffer(ModelBufferType::AttentionScore, score);

    tensor::Tensor mha_out(base::DataType::DataTypeFp32, dim, true, allocator);
    insert_buffer(ModelBufferType::MHAOutput, mha_out);

    tensor::Tensor attention_out(base::DataType::DataTypeFp32, hidden_dim, true, allocator);
    insert_buffer(ModelBufferType::AttentionOuput, attention_out);

    tensor::Tensor w1_output(base::DataType::DataTypeFp32, immediate_dim, true, allocator);
    tensor::Tensor w3_output(base::DataType::DataTypeFp32, immediate_dim, true, allocator);
    insert_buffer(ModelBufferType::FFNW1Output, w1_output);
    insert_buffer(ModelBufferType::FFNW3Output, w3_output);

    tensor::Tensor logits(base::DataType::DataTypeFp32, vocab_size, true, allocator);
    insert_buffer(ModelBufferType::Logits, logits);
}

void Qwen3Model::attention_rmsnorm(int32_t layer_id, const tensor::Tensor& input) const {
    const auto& mha_rmsnorm = qwen3_layers_->rmsnorm_layers_.at(layer_id);
    STATUS_CHECK(mha_rmsnorm->forward(input, get_buffer(ModelBufferType::MHAPreRMSNorm)));
}

void Qwen3Model::attention_qkv_rope(int32_t layer_id, const tensor::Tensor& token_pos) const {
    int32_t dim = config_->dim;
    int32_t kv_dim = config_->kv_dim;
    int32_t head_dim = config_->head_dim;
    int32_t head_num = config_->head_num;
    int32_t layer_num = config_->layer_num;
    int32_t hidden_dim = config_->hidden_dim;
    int32_t kv_head_num = config_->kv_head_num;

    // 1. KV Cache 切片 (Zero-Copy 优化): 没有申请新内存，而是去 KV Cache 显存池里，找到了当前这个 Token 应该存放的位置
    tensor::Tensor query = get_buffer(ModelBufferType::Query);
    auto [key, value] = slice_kv_cache(layer_id, token_pos.index<int32_t>(0));
    
    // 2. 线性投影: 把归一化后的数据乘以 wq, wk, wv 矩阵，得到 Query, Key, Value 向量
    const auto& wq_layer = qwen3_layers_->wq_layers_.at(layer_id);
    const auto& wk_layer = qwen3_layers_->wk_layers_.at(layer_id);
    const auto& wv_layer = qwen3_layers_->wv_layers_.at(layer_id);

    const auto& q_norm = qwen3_layers_->rmsnorm_layers_.at(layer_id + 2 * layer_num + 1);
    const auto& k_norm = qwen3_layers_->rmsnorm_layers_.at(layer_id + 3 * layer_num + 1);
    
    const tensor::Tensor& input = get_buffer(ModelBufferType::MHAPreRMSNorm);
    STATUS_CHECK(wq_layer->forward(input, query));
    query.reshape({head_num, head_dim});
    STATUS_CHECK(q_norm->forward(query, query));
    query.reshape({dim});

    STATUS_CHECK(wk_layer->forward(input, key));
    key.reshape({kv_head_num, head_dim});
    STATUS_CHECK(k_norm->forward(key, key));
    key.reshape({kv_dim});

    STATUS_CHECK(wv_layer->forward(input, value));

    // 3. RoPE 旋转位置编码: 给 Query 和 Key 向量加上位置信息，利用初始化时预计算好的 Sin/Cos 表，对向量进行旋转变换
    const tensor::Tensor& sin_cache = get_buffer(ModelBufferType::SinCache);
    const tensor::Tensor& cos_cache = get_buffer(ModelBufferType::CosCache);
    STATUS_CHECK(qwen3_layers_->rope_layer_->forward(query, key, token_pos, sin_cache, cos_cache, tensor::Tensor()));
}

void Qwen3Model::attention_mha(int32_t layer_id, const tensor::Tensor& token_pos) const {
    // 1. 获取全量 KV 缓存: 取出 key_cache 和 val_cache，包含了之前所有轮次的对话历史
    const tensor::Tensor& key_cache = get_buffer(ModelBufferType::KeyCache);
    const tensor::Tensor& value_cache = get_buffer(ModelBufferType::ValueCache);

    // 2. 设置位置 token pos + 第几层 layer id
    op::MultiHeadAttention* mha_layer_ptr = dynamic_cast<op::MultiHeadAttention*>(qwen3_layers_->mha_layer_.get());
    mha_layer_ptr->set_pos(token_pos.index<int32_t>(0));
    mha_layer_ptr->set_layer_id(layer_id);

    // 3. 计算 MHA: Attention(Q, K, V) = (softmax QK^T / sqrt(d)) V
    const tensor::Tensor& query = get_buffer(ModelBufferType::Query);
    const tensor::Tensor& score = get_buffer(ModelBufferType::AttentionScore);
    const tensor::Tensor& output = get_buffer(ModelBufferType::MHAOutput);
    STATUS_CHECK(qwen3_layers_->mha_layer_->forward(query, score, key_cache, value_cache, output));

    // 4. 输出投影: 计算出的结果再经过一个 wo (Output Weight) 线性层
    const auto& wo_layer = qwen3_layers_->wo_layers_.at(layer_id);
    STATUS_CHECK(wo_layer->forward(output, get_buffer(ModelBufferType::AttentionOuput)));
}

void Qwen3Model::feed_forward(int32_t layer_id, const tensor::Tensor& input) const {
    // 1. 残差连接 (Residual Add): 把刚才 Attention 的结果加回到原始 input 上
    STATUS_CHECK(qwen3_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::AttentionOuput), input));

    // 2. FFN 归一化: 对加完的数据再做一次 RMSNorm
    const auto& ffn_rmsnorm = qwen3_layers_->rmsnorm_layers_.at(config_->layer_num + layer_id);
    const tensor::Tensor& output = get_buffer(ModelBufferType::FFNPreRMSNorm);
    STATUS_CHECK(ffn_rmsnorm->forward(input, output));

    // 3. SwiGLU 计算: W1(Gate-proj)->w1_output，W3(Up-proj)->w3_output，W2(Down-proj)->w2_output
    const auto& w1_layer = qwen3_layers_->w1_layers_.at(layer_id);
    const auto& w2_layer = qwen3_layers_->w2_layers_.at(layer_id);
    const auto& w3_layer = qwen3_layers_->w3_layers_.at(layer_id);
    const tensor::Tensor& w1_output = get_buffer(ModelBufferType::FFNW1Output);
    const tensor::Tensor& w2_output = get_buffer(ModelBufferType::FFNW2Output);
    const tensor::Tensor& w3_output = get_buffer(ModelBufferType::FFNW3Output);
    STATUS_CHECK(w1_layer->forward(output, w1_output));
    STATUS_CHECK(w3_layer->forward(output, w3_output));
    STATUS_CHECK(qwen3_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));
    STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

    // 4. 残差连接 (Residual Add): 再次把 FFN 的结果加回 input
    STATUS_CHECK(qwen3_layers_->add_layer_->forward(input, w2_output, input));
}

void Qwen3Model::cls_logits(const tensor::Tensor& input) const {
    // 1. 最后再做一次 Final RMSNorm
    const auto& final_rmsnorm = qwen3_layers_->rmsnorm_layers_.at(2 * config_->layer_num);
    STATUS_CHECK(final_rmsnorm->forward(input, input));
    
    // 2. 将向量映射到词表大小，输出叫 Logits (未归一化的概率分值)
    const tensor::Tensor& logits = get_buffer(ModelBufferType::Logits);
    STATUS_CHECK(qwen3_layers_->cls_layer_->forward(input, logits));
}

int32_t Qwen3Model::post_process(bool is_prompt) const {
    // 1. Prompt 阶段：通常不需要采样，返回 -1，因为我们只关心读入用户的 Prompt 产生的 KV Cache，不生成
    if (is_prompt) {
        return -1;
    }
    // 2. 生成阶段：调用 sampler_->sample，根据策略 argmax 选最大的挑出下一个 token id
    const tensor::Tensor& logits = get_buffer(ModelBufferType::Logits);
    CHECK_EQ(logits.size(), config_->vocab_size);
    int32_t next_token_id = sampler_->sample(logits.ptr<float>(), logits.size(), cuda_config_ ? cuda_config_->stream : nullptr);
    return next_token_id;
}
}  // namespace model

#endif