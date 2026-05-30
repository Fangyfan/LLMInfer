#include <chrono>
#include <iostream>
#include <vector>
#include <cstdint>

#include <cuda_runtime_api.h>

#include <glog/logging.h>
#include "model/qwen3.h"
#include "model/qwen3_fused.h"

#define CUDA_CHECK(expr)                                                                    \
    do {                                                                                    \
        cudaError_t err = (expr);                                                           \
        LOG_IF(FATAL, err != cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);     \
    } while (0)

static inline void sync_cuda() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Model>
std::pair<int32_t, int32_t> generate(const Model& model, const std::string& sentence, int total_steps, double& TTFT, double& TPOT, bool need_output = false) {
    using Clock = std::chrono::steady_clock;

    TTFT = 0.0;
    TPOT = 0.0;

    std::vector<int32_t> input_token_ids = model.encode(sentence);
    LOG_IF(FATAL, input_token_ids.empty()) << "input token ids is empty!" << std::endl;

    const int32_t prompt_token_count = static_cast<int32_t>(input_token_ids.size());

    const op::EmbeddingResult& prompt_embedding_result = model.embedding(input_token_ids);

    bool is_prompt = true;
    tensor::Tensor token_pos = model.get_buffer(model::ModelBufferType::TokenPosition);

    int32_t pos = 0;
    int32_t next_token_id = input_token_ids.at(pos);

    std::vector<int32_t> generated_token_ids;

    int32_t decode_token_count = 0;

    bool first_token_generated = false;
    Clock::time_point infer_start_time;
    Clock::time_point first_token_time;
    Clock::time_point last_token_time;

    // 清理之前可能残留的 CUDA 异步任务，避免污染本次计时。
    sync_cuda();

    // benchmark 通常不把 tokenizer / encode 时间算入 TTFT。
    // 如果你想测端到端 TTFT，可以把这行移动到 model.encode(sentence) 之前。
    infer_start_time = Clock::now();

    while (pos < total_steps) {
        token_pos.index<int32_t>(0) = pos;

        if (pos < prompt_token_count - 1) {
            // Prefill 阶段：处理 prompt。
            // 不要每一步都 sync，否则会人为拉高 TTFT。
            const tensor::Tensor& token_embedding = model.get_embedding(token_pos, prompt_embedding_result, is_prompt);
            STATUS_CHECK(model.predict(token_embedding, token_pos, is_prompt, next_token_id));
        } else {
            // Decode 阶段：开始生成新 token。
            is_prompt = false;
            std::vector<int32_t> token_ids{next_token_id};
            const op::EmbeddingResult& token_embedding_result = model.embedding(token_ids);
            const tensor::Tensor& token_embedding = model.get_embedding(token_pos, token_embedding_result, is_prompt);
            STATUS_CHECK(model.predict(token_embedding, token_pos, is_prompt, next_token_id));

            // CUDA 是异步执行的。这里必须同步，否则 chrono 测到的是 launch 时间。
            sync_cuda();

            auto now = Clock::now();
            decode_token_count++;
            if (!first_token_generated) {
                first_token_generated = true;
                first_token_time = now;
            }
            last_token_time = now;

            const bool is_special = (next_token_id == 151645 || next_token_id == 151644);
            const bool is_end = model.is_sentence_end(next_token_id);
            if (!is_special && !is_end) {
                generated_token_ids.push_back(next_token_id);
            }
            if (is_end) {
                break;
            }
        }

        // 只有 prompt 阶段才从 input_token_ids 里取下一个 token。
        // decode 阶段的 next_token_id 来自 model.predict。
        if (is_prompt) {
            next_token_id = input_token_ids.at(pos + 1);
        }
        pos += 1;
    }

    sync_cuda();

    if (first_token_generated) {
        TTFT = std::chrono::duration<double>(first_token_time - infer_start_time).count();
    }

    // TPOT 通常不包含第一个 token，因为第一个 token 的耗时已经计入 TTFT。
    if (decode_token_count > 1) {
        TPOT = std::chrono::duration<double>(last_token_time - first_token_time).count() / static_cast<double>(decode_token_count - 1);
    }

    if (need_output) {
        std::cout << model.decode(generated_token_ids) << std::endl;
    }

    // 返回生成阶段 token 数更适合算 decode 性能。
    return std::make_pair(prompt_token_count, decode_token_count);
}

std::string fill_template(const std::string& prompt) {
    const std::string format = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
    std::string sentence = format;
    size_t pos = sentence.find("%s");
    if (pos != std::string::npos) {
        sentence.replace(pos, 2, prompt);
    }
    return sentence;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Please use: ./qwen3_infer checkpoint_path(.bin) tokenizer_path(.model)"
                  << std::endl;
        return -1;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];

    bool is_quant_model = true;
    using ModelType = model::Qwen3FusedModel;

    ModelType model(base::TokenizerType::EncodeBpe, tokenizer_path, checkpoint_path, is_quant_model);

    base::Status status = model.init(base::DeviceType::DeviceCUDA);
    if (!status) {
        LOG(FATAL) << "The model init failed, the error code is: " << status.get_err_code() << std::endl;
    }

    std::string prompt = "What is AI?";
    std::cout << prompt << std::endl;
    const std::string sentence = fill_template(prompt);

    std::cout << "Qwen3" << (is_quant_model ? "-AWQ-INT4" : "") << " model warmup..." << std::endl;
    {
        double warmup_TTFT = 0.0;
        double warmup_TPOT = 0.0;

        // warmup：避免首次 CUDA 初始化、kernel 加载、显存分配污染正式结果。
        // 如果你的 model 有 reset_kv_cache / clear_kv_cache 接口，建议 warmup 后调用。
        generate<ModelType>(model, sentence, 128, warmup_TTFT, warmup_TPOT, false);
    }

    std::cout << "Qwen3" << (is_quant_model ? "-AWQ-INT4" : "") << " model generating..." << std::endl;

    double TTFT = 0.0;
    double TPOT = 0.0;
    const int32_t total_steps = 4096;

    sync_cuda();
    auto start = std::chrono::steady_clock::now();

    auto [prompt_tokens, decode_tokens] = generate<ModelType>(model, sentence, total_steps, TTFT, TPOT, true);

    sync_cuda();
    auto end = std::chrono::steady_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();

    std::cout << "\n--------------- Performance Metrics ---------------" << std::endl;
    std::cout << "prompt_tokens: " << prompt_tokens << std::endl;
    std::cout << "decode_tokens: " << decode_tokens << std::endl;
    std::cout << "time(s): " << duration << std::endl;
    if (duration > 0.0) {
        std::cout << "decode_tokens/s_total: " << static_cast<double>(decode_tokens) / duration << std::endl;
    }
    std::cout << "TTFT (First Token Latency): " << TTFT * 1000.0 << " ms" << std::endl;
    std::cout << "TPOT (Average Token Latency): " << TPOT * 1000.0 << " ms" << std::endl;
    if (TPOT > 0.0) {
        std::cout << "decode_tokens/s_after_first: " << 1.0 / TPOT << std::endl;
    }
    std::cout << "---------------------------------------------------" << std::endl;

    return 0;
}