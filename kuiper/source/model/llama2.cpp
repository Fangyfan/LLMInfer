#include "model/llama2.h"
#include "op/add.h"
#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"
#include "op/rope.h"
#include "op/swiglu.h"
#include "../op/kernel/kernel_interface.h"

namespace model {
void Llama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> cuda_config) {
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

Llama2Model::Llama2Model(base::TokenizerType tokenizer_type, std::string tokenizer_path, std::string model_path, bool is_quant_model)
: Model(tokenizer_type, base::ModelType::ModelTypeLlama2, std::move(tokenizer_path), std::move(model_path), is_quant_model) {}

base::Status Llama2Model::init(base::DeviceType device_type) {
    // 1. 设备检查与环境搭建: token path 检查，CPU 量化检查, CUDA 初始化
    if (tokenizer_path_.empty()) {
        return base::error::path_not_valid(tokenizer_path_);
    }
    if (device_type == base::DeviceType::DeviceCPU && is_quant_model_) {
        return base::error::invalid_argument("The cpu device do not support int8 quant model.");
    }
    
    device_type_ = device_type;
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        if (cudaSetDevice(0) != cudaSuccess) {
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

base::Status Llama2Model::predict(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos, bool is_prompt, int32_t& next_token_id) const {
    base::Status status = forward(token_embedding, token_pos);
    if (!status) {
        return status;
    }
    next_token_id = post_process(is_prompt);
    return base::error::success();
}

base::Status Llama2Model::forward(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos) const {
    if (token_embedding.is_empty() || token_pos.is_empty()) {
        return base::error::invalid_argument("The tensor parameter in Llama2Model::forward is empty.");
    }
    if (token_embedding.get_dim(0) != config_->dim || token_pos.get_dim(0) == 1) {
        return base::error::invalid_argument("The tensor parameter in Llama2Model::forward has a wrong dim");
    }
    // 遍历所有 Transformer Block
    for (int32_t layer_id = 0; layer_id < config_->layer_num; layer_id++) {
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

op::EmbeddingResult Llama2Model::embedding(const std::vector<int32_t>& token_ids) const {
    // 1. 获取 Buffer: 从之前准备好的资源池里拿出 TokenIds 和 TokenEmbeddings
    tensor::Tensor token_ids_ = get_buffer(model::ModelBufferType::TokenIds);
    tensor::Tensor token_embeddings_ = get_buffer(model::ModelBufferType::TokenEmbeddings);

    // 2. 动态 Reshape: 虽然 Buffer 是预分配的，但预填充阶段和生成阶段的 token num 不同
    // reshape 操作通常只是修改 Tensor 的元数据（维度信息），只要新大小不超过预分配的容量，就不会触发昂贵的内存重分配
    int32_t size = static_cast<int32_t>(token_ids.size());
    if (token_ids_.size() != size) {
        token_ids_.reshape({ size });
        token_embeddings_.reshape({ size, config_->dim });
    }

    // 3. 数据填充: 把输入的 std::vector<int> 拷贝到 Buffer 中
    memcpy(token_ids_.ptr<int32_t>(), token_ids.data(), size * sizeof(int32_t));
    
    // 4. 执行查找: 本质上是查表，根据 token id 从巨大的 Embedding 矩阵中把对应的行复制出来
    tensor::Tensor token_num_(base::DataType::DataTypeInt32, size);
    STATUS_CHECK(llama2_layers_->embedding_layer_->forward(token_ids_, token_num_, token_embeddings_));
    return op::EmbeddingResult(token_ids_, token_embeddings_, token_num_);
}

base::Status Llama2Model::create_layers() {
    // 1. 先创建容器 llama2_layers_
    CHECK(llama2_layers_ == nullptr);
    llama2_layers_ = std::make_unique<Llama2Layers>();

    // 2. 创建无参数算子 create_nonparam_layers
    create_nonparam_layers();
    
    // 3. 根据是否量化，分别调用 create_param_layers 或 create_param_quant_layers
    if (!is_quant_model_) {
        create_param_layers();
    } else {
        create_param_quant_layers();
    }
    
    // 5. 算子层数量和空指针检查
    if (!llama2_layers_->embedding_layer_) {
        return base::error::internal_error("Create the embedding layer for the llama model failed!");
    }
    if (llama2_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num + 1) {
        return base::error::internal_error("Create the rmsnorm layers for the llama model failed!");
    }
    if (llama2_layers_->wq_layers_.size() != config_->layer_num ||
        llama2_layers_->wk_layers_.size() != config_->layer_num ||
        llama2_layers_->wv_layers_.size() != config_->layer_num ||
        llama2_layers_->wo_layers_.size() != config_->layer_num ||
        llama2_layers_->w1_layers_.size() != config_->layer_num ||
        llama2_layers_->w2_layers_.size() != config_->layer_num ||
        llama2_layers_->w3_layers_.size() != config_->layer_num) {
        return base::error::internal_error("Create the matmul layer in the MHA and FFN layers for llama2 model failed.");
    }
    for (int32_t i = 0; i < config_->layer_num; ++i) {
        if (!llama2_layers_->wq_layers_[i] || !llama2_layers_->wk_layers_[i] ||
            !llama2_layers_->wv_layers_[i] || !llama2_layers_->wo_layers_[i] ||
            !llama2_layers_->w1_layers_[i] || !llama2_layers_->w2_layers_[i] || !llama2_layers_->w3_layers_[i]) {
            return base::error::internal_error("Create the matmul layer in the MHA and FFN layers for llama2 model failed.");
        }
    }
    if (!llama2_layers_->mha_layer_) {
        return base::error::internal_error("Create the mha layer for the llama model failed!");
    }
    if (!llama2_layers_->rope_layer_) {
        return base::error::internal_error("Create the rope layer for the llama model failed!");
    }
    if (!llama2_layers_->swiglu_layer_) {
        return base::error::internal_error("Create the swiglu layer for the llama model failed!");
    }
    if (!llama2_layers_->add_layer_) {
        return base::error::internal_error("Create the add layer for the llama model failed!");
    }
    if (!llama2_layers_->cls_layer_) {
        return base::error::internal_error("Create the cls logits layer for the llama model failed!");
    }

    // 5. CUDA 权重搬运: 如果是 CUDA 模式，调用 llama2_layers_->to_cuda(...) 把所有权重层移到 GPU 上
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        llama2_layers_->to_cuda(cuda_config_);
    }
    return base::error::success();
}

void Llama2Model::create_nonparam_layers() {
    CHECK(llama2_layers_ != nullptr);

    int32_t dim = config_->dim;
    int32_t kv_dim = config_->kv_dim;
    int32_t kv_mul = config_->kv_mul;
    int32_t head_num = config_->head_num;
    int32_t head_dim = config_->head_dim;
    int32_t hidden_dim = config_->hidden_dim;
    int32_t max_seq_len = config_->max_seq_len;

    // 1. RoPELayer: 创建旋转位置编码算子
    // 注意参数：dim (总维度), kv_dim (KV维度，用于 GQA/MQA), head_dim (每个头的维度)
    llama2_layers_->rope_layer_ = std::make_unique<op::RoPELayer>(device_type_, dim, kv_dim, head_dim);

    // 2. MultiHeadAttention: 创建注意力算子
    // kv_mul_: 这个参数很有意思。如果 kv_mul_ > 1，说明使用了 GQA (Grouped Query Attention)，即多个 Query 头共享一组 KV 头
    // 这是 Llama 2 (70B) 和 Llama 3 的重要特性，能大幅减少显存占用
    llama2_layers_->mha_layer_ = std::make_unique<op::MultiHeadAttention>(device_type_, kv_dim, kv_mul, head_num, head_dim, max_seq_len);

    // 3. AddLayer: 向量加法算子
    // 专门用于处理残差连接（Residual Connection），即 Output = Input + F(Input)
    llama2_layers_->add_layer_ = std::make_unique<op::AddLayer>(device_type_);

    // 4. SwiGLULayer: SwiGLU 激活算子
    // 这是 Llama 相比原始 Transformer（使用 ReLU）的一大改进，能提供更好的非线性表达能力
    llama2_layers_->swiglu_layer_ = std::make_unique<op::SwiGLULayer>(device_type_, hidden_dim);
}

// 1. Embedding     : [vocab_size, dim]     : pos = 0 (开头) 读取 Embedding
// 2. RMSNorm (MHA) : [layers, dim]         : 循环读取 RMSNorm (MHA)
// 3. Attention Wq  : [layers, dim, dim]    : 循环读取 Wq
// 4. Attention Wk  : [layers, kv_dim, dim] : 循环读取 Wk
// 5. Attention Wv  : [layers, kv_dim, dim] : 循环读取 Wv
// 6. Attention Wo  : [layers, dim, dim]    : 循环读取 Wo
// 7. RMSNorm (FFN)	: [layers, dim]	        : 循环读取 RMSNorm (FFN)
// 8. FFN W1 (Gate)	: [layers, hidden, dim] : 循环读取 W1
// 9. FFN W2 (Down)	: [layers, dim, hidden] : 循环读取 W2
// 10. FFN W3 (Up)  : [layers, hidden, dim] : 循环读取 W3
// 11. FinalRMSNorm : [dim]                 : 读取 RMSNorm (Final)
// 12. RoPE Freqs   : [head/2, seq]	        : 跳过 (因为预计算)
// 13. Output Head	: [vocab_size, dim]     : 读取 CLS Layer
void Llama2Model::create_param_layers() {
    CHECK(!is_quant_model_);
    CHECK(llama2_layers_ != nullptr);
    
    size_t offset = 0;
    int32_t dim = config_->dim;
    int32_t kv_dim = config_->kv_dim;
    int32_t layer_num = config_->layer_num;
    int32_t hidden_dim = config_->hidden_dim;
    int32_t vocab_size = config_->vocab_size;
    int32_t max_seq_len = config_->max_seq_len;
    
    // 1. Embedding : [vocab_size, dim]
    llama2_layers_->embedding_layer_ = std::make_unique<op::EmbeddingLayer>(device_type_, dim, max_seq_len, vocab_size);
    llama2_layers_->embedding_layer_->set_weight(0, {vocab_size, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
    offset += vocab_size * dim;

    // 2. RMSNorm (MHA) : [layers, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto mha_rmsnorm = std::make_unique<op::RMSNormLaryer>(device_type_, dim);
        mha_rmsnorm->set_weight(0, {dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->rmsnorm_layers_.push_back(std::move(mha_rmsnorm));
        offset += dim;
    }

    // 3. Attention Wq : [layers, dim, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wq = std::make_unique<op::MatmulLayer>(device_type_, dim, dim);
        wq->set_weight(0, {dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->wq_layers_.push_back(std::move(wq));
        offset += dim * dim;
    }

    // 4. Attention Wk : [layers, kv_dim, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wk = std::make_unique<op::MatmulLayer>(device_type_, kv_dim, dim);
        wk->set_weight(0, {kv_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->wk_layers_.push_back(std::move(wk));
        offset += kv_dim * dim;
    }

    // 5. Attention Wv : [layers, kv_dim, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wv = std::make_unique<op::MatmulLayer>(device_type_, kv_dim, dim);
        wv->set_weight(0, {kv_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->wv_layers_.push_back(std::move(wv));
        offset += kv_dim * dim;
    }

    // 6. Attention Wo : [layers, dim, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wo = std::make_unique<op::MatmulLayer>(device_type_, dim, dim);
        wo->set_weight(0, {dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->wo_layers_.push_back(std::move(wo));
        offset += dim * dim;
    }

    // 7. RMSNorm (FFN)	: [layers, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto ffn_rmsnorm = std::make_unique<op::RMSNormLaryer>(device_type_, dim);
        ffn_rmsnorm->set_weight(0, {dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->rmsnorm_layers_.push_back(std::move(ffn_rmsnorm));
        offset += dim;
    }

    // 8. FFN W1 (Gate)	: [layers, hidden, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w1 = std::make_unique<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w1->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->w1_layers_.push_back(std::move(w1));
        offset += hidden_dim * dim;
    }

    // 9. FFN W2 (Down)	: [layers, dim, hidden]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w2 = std::make_unique<op::MatmulLayer>(device_type_, dim, hidden_dim);
        w2->set_weight(0, {dim, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->w2_layers_.push_back(std::move(w2));
        offset += dim * hidden_dim;
    }

    // 10. FFN W3 (Up) : [layers, hidden, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w3 = std::make_unique<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w3->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->w3_layers_.push_back(std::move(w3));
        offset += hidden_dim * dim;
    }

    // 11. Final RMSNorm : [dim]
    auto final_rmsnorm = std::make_unique<op::RMSNormLaryer>(device_type_, dim);
    final_rmsnorm->set_weight(0, {dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
    llama2_layers_->rmsnorm_layers_.push_back(std::move(final_rmsnorm));
    offset += dim;

    // 12. RoPE Sin/Cos Freqs : [head/2, seq] : skip sin/cos cache
    offset += config_->head_dim * config_->max_seq_len;

    // 13. Output Head : [vocab_size, dim]
    if (config_->is_shared_weight) offset = 0; // 权重共享: 很多小模型为了省显存，输出层和输入 Embedding 层共享同一个矩阵
    llama2_layers_->cls_layer_ = std::make_unique<op::MatmulLayer>(device_type_, vocab_size, dim);
    llama2_layers_->cls_layer_->set_weight(0, {vocab_size, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
}

// 1. Attention Layers (Wq, Wk, Wv, Wo) : Int8 + Fp32 Scales 循环读取
// 2. FFN Layers (W1, W2, W3)           : Int8 + Fp32 Scales 循环读取
// 3. CLS Layer	                        : Int8 + Fp32 Scales 单独读取
// 4. Embedding Layer	                : 纯 FP32 切换指针 单独读取
// 5. RMSNorm Layers	                : 纯 FP32 循环读取
void Llama2Model::create_param_quant_layers() {
    CHECK(is_quant_model_);
    CHECK(llama2_layers_ != nullptr);
    
    size_t offset = 0;
    int32_t dim = config_->dim;
    int32_t kv_dim = config_->kv_dim;
    int32_t layer_num = config_->layer_num;
    int32_t hidden_dim = config_->hidden_dim;
    int32_t vocab_size = config_->vocab_size;
    int32_t max_seq_len = config_->max_seq_len;
    
    // 1. Attention Wq : [layers, dim, dim] : Int8 + Fp32 Scales
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wq = std::make_unique<op::MatmulLayer>(device_type_, dim, dim, true);
        wq->set_group_size(group_size_);
        wq->set_weight(0, {dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->wq_layers_.push_back(std::move(wq));
        offset += dim * dim + wq->get_scales_size() * sizeof(float);
    }

    // 2. Attention Wk : [layers, kv_dim, dim] : Int8 + Fp32 Scales
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wk = std::make_unique<op::MatmulLayer>(device_type_, kv_dim, dim, true);
        wk->set_group_size(group_size_);
        wk->set_weight(0, {kv_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->wk_layers_.push_back(std::move(wk));
        offset += kv_dim * dim + wk->get_scales_size() * sizeof(float);
    }

    // 3. Attention Wv : [layers, kv_dim, dim] : Int8 + Fp32 Scales
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wv = std::make_unique<op::MatmulLayer>(device_type_, kv_dim, dim, true);
        wv->set_group_size(group_size_);
        wv->set_weight(0, {kv_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->wv_layers_.push_back(std::move(wv));
        offset += kv_dim * dim + wv->get_scales_size() * sizeof(float);
    }

    // 4. Attention Wo : [layers, dim, dim] : Int8 + Fp32 Scales
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wo = std::make_unique<op::MatmulLayer>(device_type_, dim, dim, true);
        wo->set_group_size(group_size_);
        wo->set_weight(0, {dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->wo_layers_.push_back(std::move(wo));
        offset += dim * dim + wo->get_scales_size() * sizeof(float);
    }

    // 5. FFN W1 (Gate)	: [layers, hidden, dim] : Int8 + Fp32 Scales
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w1 = std::make_unique<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
        w1->set_group_size(group_size_);
        w1->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->w1_layers_.push_back(std::move(w1));
        offset += hidden_dim * dim + w1->get_scales_size() * sizeof(float);
    }

    // 6. FFN W2 (Down)	: [layers, dim, hidden] : Int8 + Fp32 Scales
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w2 = std::make_unique<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
        w2->set_group_size(group_size_);
        w2->set_weight(0, {dim, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->w2_layers_.push_back(std::move(w2));
        offset += dim * hidden_dim + w2->get_scales_size() * sizeof(float);
    }

    // 7. FFN W3 (Up) : [layers, hidden, dim] : Int8 + Fp32 Scales
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w3 = std::make_unique<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
        w3->set_group_size(group_size_);
        w3->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->w3_layers_.push_back(std::move(w3));
        offset += hidden_dim * dim + w3->get_scales_size() * sizeof(float);
    }

    // 8. Output Head : [vocab_size, dim] : Int8 + Fp32 Scales
    CHECK(!config_->is_shared_weight);
    llama2_layers_->cls_layer_ = std::make_unique<op::MatmulLayer>(device_type_, vocab_size, dim, true);
    op::MatmulLayer* cls_layer_ptr = dynamic_cast<op::MatmulLayer*>(llama2_layers_->cls_layer_.get());
    cls_layer_ptr->set_group_size(group_size_);
    cls_layer_ptr->set_weight(0, {vocab_size, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
    offset += vocab_size * dim + cls_layer_ptr->get_scales_size() * sizeof(float);

    // 9. Embedding : [vocab_size, dim] : 纯 FP32
    llama2_layers_->embedding_layer_ = std::make_unique<op::EmbeddingLayer>(device_type_, dim, config_->max_seq_len, vocab_size);
    llama2_layers_->embedding_layer_->set_weight(0, {vocab_size, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
    offset += vocab_size * dim * sizeof(float);

    // 10. RMSNorm : [2 * layers + 1, dim] : 纯 FP32
    for (int32_t i = 0; i < 2 * layer_num + 1; ++i) {
        auto rmsnorm = std::make_unique<op::RMSNormLaryer>(device_type_, dim);
        rmsnorm->set_weight(0, {dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        llama2_layers_->rmsnorm_layers_.push_back(std::move(rmsnorm));
        offset += dim * sizeof(float);
    }
}

void Llama2Model::allocate_model_buffers() {
    // 1. 分配器选择: 根据配置决定是用 CPU 内存（malloc）还是 GPU 显存（cudaMalloc）
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type_ == base::DeviceType::DeviceCPU) {
        allocator = allocator_cpu;
    } else if (device_type_ == base::DeviceType::DeviceCUDA) {
        allocator = allocator_cu;
    } else {
        LOG(FATAL) << "Unknown device type in Llama2Model::allocate_model_buffers." << std::endl;
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

    tensor::Tensor token_ids(base::DataType::DataTypeInt32, 1, true, allocator_cpu);
    tensor::Tensor token_pos(base::DataType::DataTypeInt32, 1, true, allocator_cpu);
    insert_buffer(ModelBufferType::TokenIds, token_ids);
    insert_buffer(ModelBufferType::TokenPosition, token_pos);

    tensor::Tensor token_embeddings(base::DataType::DataTypeFp32, 1, dim, true, allocator);
    insert_buffer(ModelBufferType::TokenEmbeddings, token_embeddings);

    tensor::Tensor sin_cache(base::DataType::DataTypeFp32, max_seq_len, head_dim / 2, true, allocator);
    tensor::Tensor cos_cache(base::DataType::DataTypeFp32, max_seq_len, head_dim / 2, true, allocator);
    insert_buffer(ModelBufferType::SinCache, sin_cache);
    insert_buffer(ModelBufferType::CosCache, cos_cache);

    tensor::Tensor key_cache(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator);
    tensor::Tensor value_cache(base::DataType::DataTypeFp32, layer_num, max_seq_len, kv_dim, true, allocator);
    insert_buffer(ModelBufferType::KeyCache, key_cache);
    insert_buffer(ModelBufferType::ValueCache, value_cache);

    tensor::Tensor rmsnorm(base::DataType::DataTypeFp32, dim, true, allocator);
    insert_buffer(ModelBufferType::MHAPreRMSNorm, rmsnorm);
    insert_buffer(ModelBufferType::MHAOutput, rmsnorm);
    insert_buffer(ModelBufferType::FFNPreRMSNorm, rmsnorm);
    insert_buffer(ModelBufferType::FFNW2Output, rmsnorm);

    tensor::Tensor query(base::DataType::DataTypeFp32, dim, true, allocator);
    insert_buffer(ModelBufferType::Query, query);
    insert_buffer(ModelBufferType::AttentionOuput, query);

    tensor::Tensor score(base::DataType::DataTypeFp32, head_num, max_seq_len, true, allocator);
    insert_buffer(ModelBufferType::AttentionScore, score);

    tensor::Tensor w1_output(base::DataType::DataTypeFp32, hidden_dim, true, allocator);
    tensor::Tensor w3_output(base::DataType::DataTypeFp32, hidden_dim, true, allocator);
    insert_buffer(ModelBufferType::FFNW1Output, w1_output);
    insert_buffer(ModelBufferType::FFNW3Output, w3_output);

    tensor::Tensor logits(base::DataType::DataTypeFp32, vocab_size, true, allocator);
    insert_buffer(ModelBufferType::Logits, logits);
}

void Llama2Model::attention_rmsnorm(int32_t layer_id, const tensor::Tensor& input) const {
    const auto& mha_rmsnorm = llama2_layers_->rmsnorm_layers_.at(layer_id);
    STATUS_CHECK(mha_rmsnorm->forward(input, get_buffer(ModelBufferType::MHAPreRMSNorm)));
}

void Llama2Model::attention_qkv_rope(int32_t layer_id, const tensor::Tensor& token_pos) const {
    // 1. KV Cache 切片 (Zero-Copy 优化): 没有申请新内存，而是去 KV Cache 显存池里，找到了当前这个 Token 应该存放的位置
    const tensor::Tensor& query = get_buffer(ModelBufferType::Query);
    const auto& [key, value] = slice_kv_cache(layer_id, token_pos.index<int32_t>(0));
    
    // 2. 线性投影: 把归一化后的数据乘以 wq, wk, wv 矩阵，得到 Query, Key, Value 向量
    const auto& wq_layer = llama2_layers_->wq_layers_.at(layer_id);
    const auto& wk_layer = llama2_layers_->wk_layers_.at(layer_id);
    const auto& wv_layer = llama2_layers_->wv_layers_.at(layer_id);
    
    const tensor::Tensor& input = get_buffer(ModelBufferType::MHAPreRMSNorm);
    STATUS_CHECK(wq_layer->forward(input, query));
    STATUS_CHECK(wk_layer->forward(input, key));
    STATUS_CHECK(wv_layer->forward(input, value));

    // 3. RoPE 旋转位置编码: 给 Query 和 Key 向量加上位置信息，利用初始化时预计算好的 Sin/Cos 表，对向量进行旋转变换
    const tensor::Tensor& sin_cache = get_buffer(ModelBufferType::SinCache);
    const tensor::Tensor& cos_cache = get_buffer(ModelBufferType::CosCache);
    STATUS_CHECK(llama2_layers_->rope_layer_->forward(query, key, token_pos, sin_cache, cos_cache, tensor::Tensor()));
}

void Llama2Model::attention_mha(int32_t layer_id, const tensor::Tensor& token_pos) const {
    // 1. 获取全量 KV 缓存: 取出 key_cache 和 val_cache，包含了之前所有轮次的对话历史
    const tensor::Tensor& key_cache = get_buffer(ModelBufferType::KeyCache);
    const tensor::Tensor& value_cache = get_buffer(ModelBufferType::ValueCache);

    // 2. 设置位置 token pos + 第几层 layer id
    op::MultiHeadAttention* mha_layer_ptr = dynamic_cast<op::MultiHeadAttention*>(llama2_layers_->mha_layer_.get());
    mha_layer_ptr->set_pos(token_pos.index<int32_t>(0));
    mha_layer_ptr->set_layer_id(layer_id);

    // 3. 计算 MHA: Attention(Q, K, V) = (softmax QK^T / sqrt(d)) V
    const tensor::Tensor& query = get_buffer(ModelBufferType::Query);
    const tensor::Tensor& score = get_buffer(ModelBufferType::AttentionScore);
    const tensor::Tensor& output = get_buffer(ModelBufferType::MHAOutput);
    STATUS_CHECK(llama2_layers_->mha_layer_->forward(query, score, key_cache, value_cache, output));

    // 4. 输出投影: 计算出的结果再经过一个 wo (Output Weight) 线性层
    const auto& wo_layer = llama2_layers_->wo_layers_.at(layer_id);
    STATUS_CHECK(wo_layer->forward(output, get_buffer(ModelBufferType::AttentionOuput)));
}

void Llama2Model::feed_forward(int32_t layer_id, const tensor::Tensor& input) const {
    // 1. 残差连接 (Residual Add): 把刚才 Attention 的结果加回到原始 input 上
    STATUS_CHECK(llama2_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::AttentionOuput), input));

    // 2. FFN 归一化: 对加完的数据再做一次 RMSNorm
    const auto& ffn_rmsnorm = llama2_layers_->rmsnorm_layers_.at(config_->layer_num + layer_id);
    const tensor::Tensor& output = get_buffer(ModelBufferType::FFNPreRMSNorm);
    STATUS_CHECK(ffn_rmsnorm->forward(input, output));

    // 3. SwiGLU 计算: W1(Gate-proj)->w1_output，W3(Up-proj)->w3_output，W2(Down-proj)->w2_output
    const auto& w1_layer = llama2_layers_->w1_layers_.at(layer_id);
    const auto& w2_layer = llama2_layers_->w2_layers_.at(layer_id);
    const auto& w3_layer = llama2_layers_->w3_layers_.at(layer_id);
    const tensor::Tensor& w1_output = get_buffer(ModelBufferType::FFNW1Output);
    const tensor::Tensor& w2_output = get_buffer(ModelBufferType::FFNW2Output);
    const tensor::Tensor& w3_output = get_buffer(ModelBufferType::FFNW3Output);
    STATUS_CHECK(w1_layer->forward(output, w1_output));
    STATUS_CHECK(w3_layer->forward(output, w3_output));
    STATUS_CHECK(llama2_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));
    STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

    // 4. 残差连接 (Residual Add): 再次把 FFN 的结果加回 input
    STATUS_CHECK(llama2_layers_->add_layer_->forward(input, w2_output, input));
}

void Llama2Model::cls_logits(const tensor::Tensor& input) const {
    // 1. 最后再做一次 Final RMSNorm
    const auto& final_rmsnorm = llama2_layers_->rmsnorm_layers_.at(2 * config_->layer_num);
    STATUS_CHECK(final_rmsnorm->forward(input, input));
    
    // 2. 将向量映射到词表大小，输出叫 Logits (未归一化的概率分值)
    const tensor::Tensor& logits = get_buffer(ModelBufferType::Logits);
    STATUS_CHECK(llama2_layers_->cls_layer_->forward(input, logits));
}

int32_t Llama2Model::post_process(bool is_prompt) const {
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