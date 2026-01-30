#include "op/encode.h"
#include "kernel/kernel_interface.h"

#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
#include "base/unicode.h"
#include <fstream>
#endif

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

#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
// BPE编码层的正则表达式模式，用于拆分文本
const std::string PAT_STR = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

// BpeEncodeLayer 构造函数
// tokenizer_path: 分词模型文件路径
// has_bos: 是否添加起始标记
// has_eos: 是否添加结束标记
BpeEncodeLayer::BpeEncodeLayer(std::string tokenizer_path, bool has_bos, bool has_eos)
: BaseEncodeLayer(std::move(tokenizer_path), has_bos, has_eos) {
    using json = nlohmann::json;  // 使用 nlohmann JSON 库

    // 打开分词模型文件
    std::ifstream ifs(tokenizer_path_);
    // 检查文件是否成功打开
    CHECK(ifs.is_open()) << "The token model path is not valid, please check the path and type of token model.";

    json data;
    try {
        data = json::parse(ifs);  // 解析 JSON 文件
    } catch (json::parse_error&) {  // 捕获 JSON 解析异常
        LOG(FATAL) << "The token model path is not valid, please check the path and type of token model.";
    }

    // 处理特殊标记（added_tokens字段）
    const auto& datas = data["added_tokens"];
    ankerl::unordered_dense::map<std::string, int32_t> special_tokens;
    for (const auto& data_ : datas) {
        int32_t id = data_["id"];  // 获取标记 ID
        std::string content = data_["content"];  // 获取标记内容
        special_tokens.insert({content, id});  // 插入到特殊标记映射中
    }

    // 处理常规词汇表（vocab字段）
    ankerl::unordered_dense::map<std::string, int32_t> encoder;  // 存储编码器映射
    const auto& vocabs = data["model"]["vocab"];
    const auto& vocab_items = vocabs.items();  // 获取所有词汇项
    for (const auto& v : vocab_items) {
        // 将 UTF-8 字符串转换为 Unicode 码点
        const auto cpts = unicode_cpts_from_utf8(v.key());
        std::string key;
        // 将每个 Unicode 码点转换回字节表示
        for (const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            key += unicode_utf8_to_byte(utf8);
        }
        const int32_t id = v.value();  // 获取标记ID
        encoder[key] = id;  // 存储到编码器映射
    }

    // 设置特殊标记ID
    bos_id_ = special_tokens["<|begin_of_text|>"];  // 起始标记ID
    eos_id_ = special_tokens["<|end_of_text|>"];    // 结束标记ID
    stop_token1_ = eos_id_;                         // 第一个停止标记（设为结束标记）
    stop_token2_ = special_tokens["<|eot_id|>"];    // 第二个停止标记

    // 计算总标记数
    num_token_ = encoder.size() + special_tokens.size();
    // 创建 tiktoken 分词器实例
    tiktoken_ = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

// 编码函数：将文本字符串转换为标记 ID 序列
std::vector<int32_t> BpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(tiktoken_ != nullptr);  // 确保分词器已初始化

    // 准备替换映射：将空格替换为特殊字符"Ġ"
    std::map<std::string, std::string> replacements;
    replacements[" "] = "Ġ";

    // 应用替换
    std::string s = absl::StrReplaceAll(sentence, replacements);

    // 使用 tiktoken 进行编码
    auto token_ids = tiktoken_->encode(s);

    // 根据需要添加起始标记
    if (has_bos_) {
        token_ids.insert(token_ids.begin(), bos_id_);
    }
    // 根据需要添加结束标记
    if (has_eos_) {
        token_ids.push_back(eos_id_);
    }
    return token_ids;  // 返回编码后的标记 ID 序列
}

// 解码单个标记 ID（未实现）
std::string BpeEncodeLayer::decode(int32_t token_id) const {
    return "";
}

// 解码函数：将标记 ID 序列转换回文本字符串
std::string BpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(tiktoken_ != nullptr);  // 确保分词器已初始化

    // 使用 tiktoken 进行解码
    std::string s = tiktoken_->decode(token_ids);

    // 准备反向替换映射：将特殊字符"Ġ"恢复为空格
    std::map<std::string, std::string> reverse_replacements;
    reverse_replacements["Ġ"] = " ";

    // 应用反向替换
    const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
    return sentence;  // 返回解码后的文本
}

// 判断标记是否为句子结束标记
bool BpeEncodeLayer::is_sentence_end(int32_t token_id) const {
    if (token_id == stop_token1_ || token_id == stop_token2_) {
        return true;
    } else {
        return false;
    }
}

// 获取词汇表大小
int32_t BpeEncodeLayer::vocab_size() const {
    CHECK(tiktoken_ != nullptr);  // 确保分词器已初始化
    return num_token_;
}

// QwenEncodeLayer 构造函数（继承自 BpeEncodeLayer）
// Qwen 模型使用不同的特殊标记
QwenEncodeLayer::QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
: BpeEncodeLayer(std::move(token_model_path), has_bos, has_eos) {
    using json = nlohmann::json;

    // 重新打开和解析 JSON 文件（与基类类似但使用不同的特殊标记）
    std::ifstream ifs(tokenizer_path_);
    json data = json::parse(ifs);

    // 处理特殊标记（与基类相同逻辑）
    const auto& datas = data["added_tokens"];
    ankerl::unordered_dense::map<std::string, int32_t> special_tokens;
    for (const auto& data_ : datas) {
        int id = data_["id"];
        std::string content = data_["content"];
        special_tokens.insert({content, id});
    }

    // 处理常规词汇表（与基类相同逻辑）
    ankerl::unordered_dense::map<std::string, int32_t> encoder;
    const auto& vocabs = data["model"]["vocab"];
    const auto& vocab_items = vocabs.items();
    for (const auto& v : vocab_items) {
        const auto cpts = unicode_cpts_from_utf8(v.key());
        std::string key;
        for (const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            key += unicode_utf8_to_byte(utf8);
        }
        const int32_t id = v.value();
        encoder[key] = id;
    }

    // 设置 Qwen 特有的特殊标记 ID
    bos_id_ = special_tokens["<|im_start|>"];  // Qwen 起始标记
    eos_id_ = special_tokens["<|im_end|>"];    // Qwen 结束标记
    stop_token1_ = eos_id_;
    stop_token2_ = special_tokens["<|endoftext|>"];  // Qwen 文本结束标记

    // 计算总标记数
    num_token_ = encoder.size() + special_tokens.size();
    // 创建 tiktoken 分词器实例（使用相同的正则表达式模式）
    tiktoken_ = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}
#endif
}  // namespace op