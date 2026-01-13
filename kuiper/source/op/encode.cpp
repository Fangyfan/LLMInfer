#include "op/encode.h"
#include "kernel/kernel_interface.h"

namespace op {
BaseEncodeLayer::BaseEncodeLayer(std::string tokenizer_path, bool has_bos, bool has_eos)
: Layer(base::DeviceType::DeviceCPU, LayerType::LayerEncode, "Encode"), 
  tokenizer_path_(std::move(tokenizer_path)), has_bos_(has_bos), has_eos_(has_eos) {}


SpeEncodeLayer::SpeEncodeLayer(std::string tokenizer_path, bool has_bos, bool has_eos)
: BaseEncodeLayer(std::move(tokenizer_path), has_bos, has_eos) {
    // 创建 processor
    spe_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    // 加载模型
    sentencepiece::util::Status status = spe_->Load(tokenizer_path_);
    // 校验是否成功，失败直接终止程序（不是异常，是 fatal error）
    if (status.code() != sentencepiece::util::StatusCode::kOk) {
        LOG(FATAL) << "The token model path is not valid, please check the path and type of token model." << std::endl;
    }
}

std::vector<int32_t> SpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(spe_ != nullptr);
    
    // SentencePiece 编码 (Unicode 归一化 -> 子词切分 -> 查表映射为 id)
    // "Hello world" -> [15043, 2789]
    std::vector<int32_t> token_ids = spe_->EncodeAsIds(sentence);

    if (has_bos_) { // 插入 BOS: [BOS, 15043, 2789]
        token_ids.insert(token_ids.begin(), spe_->bos_id());
    }
    if (has_eos_) { // 插入 EOS: [BOS, 15043, 2789, EOS]
        token_ids.push_back(spe_->eos_id());
    }
    return token_ids;
}

std::string SpeEncodeLayer::decode(int32_t token_id) const {
    CHECK(spe_ != nullptr);
    return spe_->DecodeIds(std::vector<int32_t>{ token_id });
}

std::string SpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(spe_ != nullptr);
    return spe_->DecodeIds(token_ids);
}

bool SpeEncodeLayer::is_sentence_end(int32_t token_id) const {
    CHECK(spe_ != nullptr);
    return token_id == spe_->eos_id();
}

int32_t SpeEncodeLayer::vocab_size() const {
    CHECK(spe_ != nullptr);
    return spe_->GetPieceSize();
}
}  // namespace op