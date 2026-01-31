#include <chrono>
#include <iostream>
#include <glog/logging.h>
#include "model/qwen3.h"

int32_t generate(const model::Qwen3Model& model, const std::string& sentence, int total_steps, bool need_output = false) {
    std::vector<int32_t> input_token_ids = model.encode(sentence);
    LOG_IF(FATAL, input_token_ids.empty()) << "input token ids is empty!" << std::endl;

    int32_t prompt_len = static_cast<int32_t>(input_token_ids.size());
    const op::EmbeddingResult& prompt_embedding_result = model.embedding(input_token_ids);
    
    bool is_prompt = true;
    tensor::Tensor token_pos = model.get_buffer(model::ModelBufferType::TokenPosition);
    int32_t pos = 0;
    int32_t next_token_id = input_token_ids.at(pos);
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
            if (next_token_id != 151645 && next_token_id != 151644) {
                generated_token_ids.push_back(next_token_id);
            }
        }
        if (model.is_sentence_end(next_token_id)) {
            break;
        }
        if (is_prompt) {
            next_token_id = input_token_ids.at(pos + 1);
        }
        pos += 1;
    }
    if (need_output) {
        std::cout << model.decode(generated_token_ids) << std::endl;
    }
    return pos;
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
        LOG(INFO) << "Please use: ./qwen3_infer checkpoint_path(.bin) tokenizer_path(.model)" << std::endl;
        return -1;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];

    bool is_quant_model = false;
    model::Qwen3Model model(base::TokenizerType::EncodeBpe, tokenizer_path, checkpoint_path, is_quant_model);

    base::Status status = model.init(base::DeviceType::DeviceCUDA);
    if (!status) {
        LOG(FATAL) << "The model init failed, the error code is: " << status.get_err_code() << std::endl;
    }

    std::string prompt = "What is AI?";
    std::cout << prompt << std::endl;
    const std::string& sentence = fill_template(prompt);

    auto start = std::chrono::steady_clock().now();
    std::cout << "Qwen3" << (is_quant_model ? "-quant8" : "") << " model generating..." << std::endl;

    const int32_t total_steps = 2560;
    int32_t steps = generate(model, sentence, total_steps, true);

    auto end = std::chrono::steady_clock().now();
    auto duration = std::chrono::duration<double>(end - start).count();
    std::cout << "steps: " << steps << std::endl;
    std::cout << "time(s): " << duration << std::endl;
    std::cout << "steps/s: " << steps / duration << std::endl;

    return 0;
}