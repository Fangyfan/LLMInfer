#include "model/qwen3_fused.h"
#include "op/add.h"
#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"
#include "op/rope.h"
#include "op/swiglu.h"
#include "op/qkv_matmul.h"
#include "op/qk_norm_rope.h"
#include "op/gate_up_swiglu.h"
#include "op/swiglu.h"
#include "../op/kernel/kernel_interface.h"

namespace model {
void Qwen3FusedLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> cuda_config) {
    CHECK_NE(cuda_config, nullptr);
    lm_head_layer_->set_cuda_config(cuda_config);
    embedding_layer_->set_cuda_config(cuda_config);
    final_rmsnorm_layer_->set_cuda_config(cuda_config);
    flashdecoding_gqa_layer_->set_cuda_config(cuda_config);

    lm_head_layer_->to_cuda();
    embedding_layer_->to_cuda();
    final_rmsnorm_layer_->to_cuda();
    flashdecoding_gqa_layer_->to_cuda();

    for (auto& pre_rmsnorm_layer : pre_rmsnorm_layers_) {
        pre_rmsnorm_layer->set_cuda_config(cuda_config);
        pre_rmsnorm_layer->to_cuda();
    }
    for (auto& fused_qkv_proj_layer : fused_qkv_proj_layers_) {
        fused_qkv_proj_layer->set_cuda_config(cuda_config);
        fused_qkv_proj_layer->to_cuda();
    }
    for (auto& fused_qk_norm_rope_layer_ : fused_qk_norm_rope_layers_) {
        fused_qk_norm_rope_layer_->set_cuda_config(cuda_config);
        fused_qk_norm_rope_layer_->to_cuda();
    }
    for (auto& fused_o_proj_add_layer : fused_o_proj_add_layers_) {
        fused_o_proj_add_layer->set_cuda_config(cuda_config);
        fused_o_proj_add_layer->to_cuda();
    }
    for (auto& ffn_rmsnorm_layer : ffn_rmsnorm_layers_) {
        ffn_rmsnorm_layer->set_cuda_config(cuda_config);
        ffn_rmsnorm_layer->to_cuda();
    }
    for (auto& fused_gate_up_swiglu_layer_ : fused_gate_up_swiglu_layers_) {
        fused_gate_up_swiglu_layer_->set_cuda_config(cuda_config);
        fused_gate_up_swiglu_layer_->to_cuda();
    }
    for (auto& fused_down_add_layer_ : fused_down_proj_add_layers_) {
        fused_down_add_layer_->set_cuda_config(cuda_config);
        fused_down_add_layer_->to_cuda();
    }
}

Qwen3FusedModel::Qwen3FusedModel(base::TokenizerType tokenizer_type, std::string tokenizer_path, std::string model_path, bool is_quant_model)
: Model(tokenizer_type, base::ModelType::ModelTypeQwen3, std::move(tokenizer_path), std::move(model_path), is_quant_model) {}

base::Status Qwen3FusedModel::init(base::DeviceType device_type) {
    // 1. 设备检查与环境搭建: token path 检查，CPU 量化检查, CUDA 初始化
    if (tokenizer_path_.empty()) {
        return base::error::path_not_valid(tokenizer_path_);
    }
    if (is_quant_model_) {
        return base::error::invalid_argument("Qwen3 do not support int8 quant model.");
    }
    
    device_type_ = device_type;
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        if (cudaSetDevice(1) != cudaSuccess) {
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

base::Status Qwen3FusedModel::predict(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos, bool is_prompt, int32_t& next_token_id) const {
    base::Status status = forward(token_embedding, token_pos);
    if (!status) {
        return status;
    }
    next_token_id = post_process(is_prompt);
    return base::error::success();
}

base::Status Qwen3FusedModel::forward(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos) const {
    if (token_embedding.is_empty()) {
        return base::error::invalid_argument("The token_embedding in Qwen3FusedModel::forward is empty.");
    }
    if (token_pos.is_empty()) {
        return base::error::invalid_argument("The token_pos in Qwen3FusedModel::forward is empty.");
    }
    if (token_embedding.dims_size() != 1 || token_embedding.get_dim(0) != config_->hidden_dim) {
        return base::error::invalid_argument("The token_embedding in Qwen3FusedModel::forward has a wrong dim");
    }
    if (token_pos.get_dim(0) != 1) {
        return base::error::invalid_argument("The token_pos in Qwen3FusedModel::forward has a wrong dim");
    }

    // 遍历所有 Transformer Block
    for (int32_t layer_id = 0; layer_id < config_->layer_num; ++layer_id) {
        // 1. 对输入 RMSNorm 归一化 -> 计算 QKV -> 缓存 KV -> RoPE 旋转位置编码
        rmsnorm_qkv_rope(layer_id, token_embedding, token_pos);
        // 2. 多头注意力
        flash_decoding_gqa(layer_id, token_embedding, token_pos);
        // 3. 前馈网络 (FFN)
        feed_forward(layer_id, token_embedding);
    }
    
    // 4. 最终分类
    cls_logits(token_embedding);
    return base::error::success();
}

op::EmbeddingResult Qwen3FusedModel::embedding(const std::vector<int32_t>& token_ids) const {
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
    STATUS_CHECK(qwen3_fused_layers_->embedding_layer_->forward(token_ids_, token_num_, token_embeddings_));
    CHECK(token_embeddings_.dims_size() == 2);
    return op::EmbeddingResult(token_ids_, token_embeddings_, token_num_);
}

base::Status Qwen3FusedModel::create_layers() {
    // 1. 先创建容器 qwen3_fused_layers_
    CHECK(qwen3_fused_layers_ == nullptr);
    qwen3_fused_layers_ = std::make_unique<Qwen3FusedLayers>();

    // 2. 创建无参数算子 create_nonparam_layers
    create_nonparam_layers();
    
    // 3. 根据是否量化，分别调用 create_param_layers 或 create_param_quant_layers
    if (!is_quant_model_) {
        create_param_layers();
    } else {
        return base::error::function_not_implement("");
    }
    
    // 4. 算子层数量和空指针检查
    if (!qwen3_fused_layers_->embedding_layer_) {
        return base::error::internal_error("Create the embedding layer for the qwen3 model failed!");
    }
    if (!qwen3_fused_layers_->flashdecoding_gqa_layer_) {
        return base::error::internal_error("Create the flashdecoding qga layer for the qwen3 model failed!");
    }
    if (!qwen3_fused_layers_->final_rmsnorm_layer_) {
        return base::error::internal_error("Create the final-rmsnorm layer for the qwen3 model failed!");
    }
    if (!qwen3_fused_layers_->lm_head_layer_) {
        return base::error::internal_error("Create the lm-head layer for the qwen3 model failed!");
    }
    if (qwen3_fused_layers_->pre_rmsnorm_layers_.size() != config_->layer_num) {
        return base::error::internal_error("Create the pre-rmsnorm layers for the qwen3 model failed!");
    }
    if (qwen3_fused_layers_->fused_qkv_proj_layers_.size() != config_->layer_num) {
        return base::error::internal_error("Create the fused-qkv-proj layers for the qwen3 model failed!");
    }
    if (qwen3_fused_layers_->fused_qk_norm_rope_layers_.size() != config_->layer_num) {
        return base::error::internal_error("Create the fused-qk-norm-rope layers for the qwen3 model failed!");
    }
    if (qwen3_fused_layers_->fused_o_proj_add_layers_.size() != config_->layer_num) {
        return base::error::internal_error("Create the fused-o-proj-add layers for the qwen3 model failed!");
    }
    if (qwen3_fused_layers_->ffn_rmsnorm_layers_.size() != config_->layer_num) {
        return base::error::internal_error("Create the ffn-rmsnorm layers for the qwen3 model failed!");
    }
    if (qwen3_fused_layers_->fused_gate_up_swiglu_layers_.size() != config_->layer_num) {
        return base::error::internal_error("Create the fused-gate-up-swiglu layers for the qwen3 model failed!");
    }
    if (qwen3_fused_layers_->fused_down_proj_add_layers_.size() != config_->layer_num) {
        return base::error::internal_error("Create the fused-down-add layers for the qwen3 model failed!");
    }
    for (int32_t i = 0; i < config_->layer_num; ++i) {
        if (!qwen3_fused_layers_->pre_rmsnorm_layers_[i] ||
            !qwen3_fused_layers_->fused_qkv_proj_layers_[i] ||
            !qwen3_fused_layers_->fused_qk_norm_rope_layers_[i] ||
            !qwen3_fused_layers_->fused_o_proj_add_layers_[i] ||
            !qwen3_fused_layers_->ffn_rmsnorm_layers_[i] ||
            !qwen3_fused_layers_->fused_gate_up_swiglu_layers_[i] ||
            !qwen3_fused_layers_->fused_down_proj_add_layers_[i]) {
            return base::error::internal_error("Create layer in the MHA and FFN layers for qwen3 model failed.");
        }
    }

    // 5. CUDA 权重搬运: 如果是 CUDA 模式，调用 qwen3_fused_layers_->to_cuda(...) 把所有权重层移到 GPU 上
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        qwen3_fused_layers_->to_cuda(cuda_config_);
    }
    return base::error::success();
}

void Qwen3FusedModel::create_nonparam_layers() {
    CHECK(qwen3_fused_layers_ != nullptr);
    int32_t kv_dim = config_->kv_dim;
    int32_t kv_mul = config_->kv_mul;
    int32_t head_num = config_->head_num;
    int32_t head_dim = config_->head_dim;
    int32_t max_seq_len = config_->max_seq_len;
    
    qwen3_fused_layers_->flashdecoding_gqa_layer_ = 
        std::make_unique<op::MultiHeadAttention>(device_type_, kv_dim, kv_mul, head_num, head_dim, max_seq_len);
}

void Qwen3FusedModel::create_param_layers() {
    CHECK(!is_quant_model_);
    CHECK(qwen3_fused_layers_ != nullptr);
    
    size_t offset = 0;
    int32_t dim = config_->dim;
    int32_t kv_dim = config_->kv_dim;
    int32_t head_dim = config_->head_dim;
    int32_t layer_num = config_->layer_num;
    int32_t vocab_size = config_->vocab_size;
    int32_t hidden_dim = config_->hidden_dim;
    int32_t max_seq_len = config_->max_seq_len;
    int32_t immediate_dim = config_->immediate_dim;
    
    // 1. Embedding : [vocab_size, hidden_dim]
    qwen3_fused_layers_->embedding_layer_ = std::make_unique<op::EmbeddingLayer>(device_type_, hidden_dim, max_seq_len, vocab_size);
    qwen3_fused_layers_->embedding_layer_->set_weight(0, {vocab_size, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
    offset += vocab_size * hidden_dim;

    // 2. Pre_RMSNorm (rmsnorm) : [layers, hidden_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto pre_rmsnorm = std::make_unique<op::RMSNormLaryer>(device_type_, hidden_dim);
        pre_rmsnorm->set_weight(0, {hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_fused_layers_->pre_rmsnorm_layers_.push_back(std::move(pre_rmsnorm));
        offset += hidden_dim;
    }

    // 3. Fused_QKV_Proj (w_qkv = w_q + w_k + w_v) : [layers, dim + 2 * kv_dim, hidden_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w_qkv = std::make_unique<op::QKVMatmulLayer>(device_type_, dim, kv_dim, hidden_dim);
        w_qkv->set_weight(0, {dim + 2 * kv_dim, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_fused_layers_->fused_qkv_proj_layers_.push_back(std::move(w_qkv));
        offset += (dim + 2 * kv_dim) * hidden_dim;
    }

    // 4. Fused_QK_Norm_RoPE (qk_norm = q_norm + k_norm) : [layers, 2 * head_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto qk_norm = std::make_unique<op::QKNormRoPELaryer>(device_type_, dim, kv_dim, head_dim);
        qk_norm->set_weight(0, {2 * head_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_fused_layers_->fused_qk_norm_rope_layers_.push_back(std::move(qk_norm));
        offset += 2 * head_dim;
    }

    // 5. Fused_O_Proj_Add (w_o) : [layers, hidden_dim, dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w_o = std::make_unique<op::MatmulLayer>(device_type_, hidden_dim, dim, false, true); // fuse_add = true
        w_o->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_fused_layers_->fused_o_proj_add_layers_.push_back(std::move(w_o));
        offset += hidden_dim * dim;
    }

    // 6. FFN_RMSNorm (rmsnorm) : [layers, hidden_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto ffn_rmsnorm = std::make_unique<op::RMSNormLaryer>(device_type_, hidden_dim);
        ffn_rmsnorm->set_weight(0, {hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_fused_layers_->ffn_rmsnorm_layers_.push_back(std::move(ffn_rmsnorm));
        offset += hidden_dim;
    }

    // 7. Fused_Gate_Up_Swiglu (w_gate + w_up) : [layers, 2 * immediate_dim, hidden_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w_gate_up = std::make_unique<op::GateUpSwigluLayer>(device_type_, immediate_dim, hidden_dim);
        w_gate_up->set_weight(0, {2 * immediate_dim, hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_fused_layers_->fused_gate_up_swiglu_layers_.push_back(std::move(w_gate_up));
        offset += 2 * immediate_dim * hidden_dim;
    }

    // 8. Fused_Down_Add (w_down) : [layers, hidden_dim, immediate_dim]
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w_down = std::make_unique<op::MatmulLayer>(device_type_, hidden_dim, immediate_dim, false, true); // fuse_add = true
        w_down->set_weight(0, {hidden_dim, immediate_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
        qwen3_fused_layers_->fused_down_proj_add_layers_.push_back(std::move(w_down));
        offset += hidden_dim * immediate_dim;
    }

    // 9. Final_RMSNorm : [hidden_dim]
    qwen3_fused_layers_->final_rmsnorm_layer_ = std::make_unique<op::RMSNormLaryer>(device_type_, hidden_dim);
    qwen3_fused_layers_->final_rmsnorm_layer_->set_weight(0, {hidden_dim}, raw_model_data_->weight_ptr(offset), base::DeviceType::DeviceCPU);
    offset += hidden_dim;

    // 10. LM_Head : [vocab_size, hidden_dim] and tie_word_embedding = true
    qwen3_fused_layers_->lm_head_layer_ = std::make_unique<op::MatmulLayer>(device_type_, vocab_size, hidden_dim, true); // lm_head = true
    qwen3_fused_layers_->lm_head_layer_->set_weight(0, {vocab_size, hidden_dim}, raw_model_data_->weight_ptr(0), base::DeviceType::DeviceCPU);
}

void Qwen3FusedModel::create_param_quant_layers() {}

void Qwen3FusedModel::allocate_model_buffers() {
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto allocator_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        allocator = allocator_cu;
    } else if (device_type_ == base::DeviceType::DeviceCPU) {
        allocator = allocator_cpu;
    } else {
        LOG(FATAL) << "Unknown device type in Qwen3FusedModel::allocate_model_buffers." << std::endl;
    }

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

    tensor::Tensor token_embeddings(base::DataType::DataTypeBf16, 1, hidden_dim, true, allocator);
    insert_buffer(ModelBufferType::TokenEmbeddings, token_embeddings);

    tensor::Tensor sin_cache(base::DataType::DataTypeFp32, max_seq_len, head_dim / 2, true, allocator);
    tensor::Tensor cos_cache(base::DataType::DataTypeFp32, max_seq_len, head_dim / 2, true, allocator);
    insert_buffer(ModelBufferType::SinCache, sin_cache);
    insert_buffer(ModelBufferType::CosCache, cos_cache);

    tensor::Tensor key_cache(base::DataType::DataTypeBf16, layer_num, max_seq_len, kv_dim, true, allocator);
    tensor::Tensor value_cache(base::DataType::DataTypeBf16, layer_num, max_seq_len, kv_dim, true, allocator);
    insert_buffer(ModelBufferType::KeyCache, key_cache);
    insert_buffer(ModelBufferType::ValueCache, value_cache);

    tensor::Tensor norm_input(base::DataType::DataTypeBf16, hidden_dim, true, allocator);
    insert_buffer(ModelBufferType::MHAPreRMSNorm, norm_input);
    insert_buffer(ModelBufferType::FFNPreRMSNorm, norm_input);

    tensor::Tensor query(base::DataType::DataTypeBf16, dim, true, allocator);
    tensor::Tensor gqa_output(base::DataType::DataTypeBf16, dim, true, allocator);
    insert_buffer(ModelBufferType::Query, query);
    insert_buffer(ModelBufferType::MHAOutput, gqa_output);

    tensor::Tensor kv_split_output(base::DataType::DataTypeFp32, head_num, 4 * max_seq_len, true, allocator);
    insert_buffer(ModelBufferType::KVSplitOutput, kv_split_output);

    tensor::Tensor swiglu_output(base::DataType::DataTypeBf16, immediate_dim, true, allocator);
    insert_buffer(ModelBufferType::SwiGLUOutput, swiglu_output);

    tensor::Tensor logits(base::DataType::DataTypeFp32, vocab_size, true, allocator);
    insert_buffer(ModelBufferType::Logits, logits);

    tensor::Tensor argmax_token(base::DataType::DataTypeInt32, 1, true, allocator);
    tensor::Tensor argmax_buffer(base::DataType::DataTypeInt32, 128 * 2, true, allocator);
    insert_buffer(ModelBufferType::ArgmaxToken, argmax_token);
    insert_buffer(ModelBufferType::ArgmaxBuffer, argmax_buffer);
}

void Qwen3FusedModel::rmsnorm_qkv_rope(int32_t layer_id, const tensor::Tensor& input, const tensor::Tensor& token_pos) const {
    // 1. Attention Pre RMSNorm
    const tensor::Tensor& norm_input = get_buffer(ModelBufferType::MHAPreRMSNorm);
    STATUS_CHECK(qwen3_fused_layers_->pre_rmsnorm_layers_.at(layer_id)->forward(input, norm_input));

    // 2. Fused QKV Proj => [qeury, key, value]
    tensor::Tensor query = get_buffer(ModelBufferType::Query);
    const auto& [key, value] = slice_kv_cache(layer_id, token_pos.index<int32_t>(0));
    STATUS_CHECK(qwen3_fused_layers_->fused_qkv_proj_layers_.at(layer_id)->
        forward(norm_input, query, key, value, tensor::Tensor()));

    // 3. QK-Norm + QK-RoPE
    const tensor::Tensor& sin_cache = get_buffer(ModelBufferType::SinCache);
    const tensor::Tensor& cos_cache = get_buffer(ModelBufferType::CosCache);
    STATUS_CHECK(qwen3_fused_layers_->fused_qk_norm_rope_layers_.at(layer_id)->
        forward(query, key, token_pos, sin_cache, cos_cache, tensor::Tensor()));
}

void Qwen3FusedModel::flash_decoding_gqa(int32_t layer_id, const tensor::Tensor& residual_add, const tensor::Tensor& token_pos) const {
    // 1. Set Token Pos + Layer Id
    op::MultiHeadAttention* gqa_layer_ptr = 
        dynamic_cast<op::MultiHeadAttention*>(qwen3_fused_layers_->flashdecoding_gqa_layer_.get());
    gqa_layer_ptr->set_pos(token_pos.index<int32_t>(0));
    gqa_layer_ptr->set_layer_id(layer_id);

    // 2. Attention GQA(Q, K, V) = (softmax QK^T / sqrt(d)) V
    const tensor::Tensor& query = get_buffer(ModelBufferType::Query);
    const tensor::Tensor& key_cache = get_buffer(ModelBufferType::KeyCache);
    const tensor::Tensor& value_cache = get_buffer(ModelBufferType::ValueCache);
    const tensor::Tensor& kv_split_output = get_buffer(ModelBufferType::KVSplitOutput);
    const tensor::Tensor& gqa_output = get_buffer(ModelBufferType::MHAOutput);
    STATUS_CHECK(qwen3_fused_layers_->flashdecoding_gqa_layer_->
        forward(query, key_cache, value_cache, kv_split_output, gqa_output));

    // 3. O Proj + Residual Add
    STATUS_CHECK(qwen3_fused_layers_->fused_o_proj_add_layers_.at(layer_id)->forward(gqa_output, residual_add));
}

void Qwen3FusedModel::feed_forward(int32_t layer_id, const tensor::Tensor& residual_add) const {
    // 1. FFN Pre RMSNorm
    const tensor::Tensor& norm_input = get_buffer(ModelBufferType::FFNPreRMSNorm);
    STATUS_CHECK(qwen3_fused_layers_->ffn_rmsnorm_layers_.at(layer_id)->forward(residual_add, norm_input));
    
    // 2. Gate Proj + Up Proj + SwiGLU
    const tensor::Tensor& swiglu_output = get_buffer(ModelBufferType::SwiGLUOutput);
    STATUS_CHECK(qwen3_fused_layers_->fused_gate_up_swiglu_layers_.at(layer_id)->forward(norm_input, swiglu_output));

    // 3. Down Proj + Residual Add
    STATUS_CHECK(qwen3_fused_layers_->fused_down_proj_add_layers_.at(layer_id)->forward(swiglu_output, residual_add));
}

void Qwen3FusedModel::cls_logits(const tensor::Tensor& input) const {
    // 1. Final RMSNorm
    STATUS_CHECK(qwen3_fused_layers_->final_rmsnorm_layer_->forward(input, input));
    
    // 2. LM Head Linear => Logits
    const tensor::Tensor& logits = get_buffer(ModelBufferType::Logits);
    STATUS_CHECK(qwen3_fused_layers_->lm_head_layer_->forward(input, logits));
}

int32_t Qwen3FusedModel::post_process(bool is_prompt) const {
    // 1. Prompt 阶段：通常不需要采样，返回 -1，因为我们只关心读入用户的 Prompt 产生的 KV Cache，不生成
    if (is_prompt) {
        return -1;
    }
    // 2. 生成阶段：调用 sampler_->sample，根据策略 argmax 选最大的挑出下一个 token id
    const tensor::Tensor& logits = get_buffer(ModelBufferType::Logits);
    const tensor::Tensor& argmax_token = get_buffer(ModelBufferType::ArgmaxToken);
    const tensor::Tensor& argmax_buffer = get_buffer(ModelBufferType::ArgmaxBuffer);
    CHECK_EQ(logits.size(), config_->vocab_size);
    int32_t next_token_id = sampler_->sample(
        logits.ptr<float>(), 
        logits.size(), 
        const_cast<int32_t*>(argmax_token.ptr<int32_t>()), 
        const_cast<void*>(argmax_buffer.ptr<void>()), 
        cuda_config_ ? cuda_config_->stream : nullptr
    );
    return next_token_id;
}

}  // namespace model