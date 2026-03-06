#include <chrono>
#include <iostream>
#include <glog/logging.h>
#include "model/llama2.h"

int32_t generate(const model::Llama2Model& model, const std::string& sentence, int total_steps, double& TTFT, double& TPOT, bool need_output = false) {
    // 在此处记录首字延迟 TTFT 的起始时间
    auto ttft_start = std::chrono::steady_clock::now();
    bool is_first_token = true; // 标记是否为第一个生成的 token

    // 用于计算平均生成延迟 TPOT 的变量
    double total_latency = 0.0; // 后续所有 token 的总耗时
    int gen_token_count = 0; // 生成的 token 总数（含第一个）
    auto last_token_time = std::chrono::steady_clock::now(); // 上一个 token 的生成时间

    std::vector<int32_t> input_token_ids = model.encode(sentence);
    LOG_IF(FATAL, input_token_ids.empty()) << "input token ids is empty!" << std::endl;

    int32_t prompt_len = static_cast<int32_t>(input_token_ids.size());
    const op::EmbeddingResult& prompt_embedding_result = model.embedding(input_token_ids);
    
    bool is_prompt = true;
    tensor::Tensor token_pos = model.get_buffer(model::ModelBufferType::TokenPosition);
    int32_t pos = 0;
    int32_t next_token_id = -1;
    // std::vector<int32_t> generated_token_ids { input_token_ids.at(0) };
    std::vector<int32_t> generated_token_ids;

    while (pos < total_steps) {
        token_pos.index<int32_t>(0) = pos;
        if (pos < prompt_len - 1) {
            const tensor::Tensor& token_embedding = model.get_embedding(token_pos, prompt_embedding_result, is_prompt);
            STATUS_CHECK(model.predict(token_embedding, token_pos, is_prompt, next_token_id));
        } else {
            is_prompt = false;
            std::vector<int32_t> token_ids { next_token_id };
            const op::EmbeddingResult& token_embedding_result = model.embedding(token_ids);
            const tensor::Tensor& token_embedding = model.get_embedding(token_pos, token_embedding_result, is_prompt);
            STATUS_CHECK(model.predict(token_embedding, token_pos, is_prompt, next_token_id));

            // 在此处计算首字延迟 TTFT (仅在生成第一个 token 时计算一次)
            if (is_first_token) {
                auto ttft_end = std::chrono::steady_clock::now();
                TTFT = std::chrono::duration<double>(ttft_end - ttft_start).count();
                is_first_token = false;

                // 初始化平均延迟相关变量
                last_token_time = ttft_end;
                gen_token_count = 1;
            } else {
                // 计算当前 token 与上一个 token 的时间差，并累加
                auto current_time = std::chrono::steady_clock::now();
                double token_latency = std::chrono::duration<double>(current_time - last_token_time).count();
                total_latency += token_latency;
                
                // 更新上一个 token 的时间和计数
                last_token_time = current_time;
                gen_token_count += 1;
            }

        }
        if (model.is_sentence_end(next_token_id)) {
            break;
        }
        if (is_prompt) {
            next_token_id = input_token_ids.at(pos + 1);
        }
        generated_token_ids.push_back(next_token_id);
        pos += 1;
    }

    // 循环结束后，计算平均生成延迟，若只生成了一个 token，平均延迟为 0（避免除以 0）
    if (gen_token_count > 1) {
        TPOT = total_latency / (gen_token_count - 1); // 排除第一个 token，取后续的平均
    }

    if (need_output) {
        std::cout << model.decode(generated_token_ids) << std::endl;
    }
    return pos;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Please use: ./llama_infer checkpoint_path(.bin) tokenizer_path(.model)" << std::endl;
        return -1;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];

    bool is_quant_model = false;
    model::Llama2Model model(base::TokenizerType::EncodeSpe, tokenizer_path, checkpoint_path, is_quant_model);

    base::Status status = model.init(base::DeviceType::DeviceCUDA);
    if (!status) {
        LOG(FATAL) << "The model init failed, the error code is: " << status.get_err_code() << std::endl;
    }

    if (model.tokenizer_type() == base::TokenizerType::EncodeSpe) {
        std::cout << "Llama2" << (is_quant_model ? "-quant8" : "") << " model generating..." << std::endl;
    } else {
        std::cout << "Llama3.2" << (is_quant_model ? "-quant8" : "") << " model generating..." << std::endl;
    }
    auto start = std::chrono::steady_clock().now();

    double TTFT = 0.0f;
    double TPOT = 0.0f;
    const std::string& sentence = "hello";
    const int32_t total_steps = 128;
    int32_t steps = generate(model, sentence, total_steps, TTFT, TPOT, true);

    auto end = std::chrono::steady_clock().now();
    auto duration = std::chrono::duration<double>(end - start).count();

    // 输出性能指标
    std::cout << "\n--------------- Performance Metrics ---------------" << std::endl;
    std::cout << "steps: " << steps << std::endl;
    std::cout << "time(s): " << duration << std::endl;
    std::cout << "steps/s: " << steps / duration << std::endl;
    std::cout << "TTFT (First Token Latency): " << TTFT * 1000 << "ms" << std::endl;
    std::cout << "TPOT (Average Token Latency): " << TPOT * 1000 << "ms" << std::endl;

    return 0;
}