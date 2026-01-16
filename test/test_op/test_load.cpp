#include "model/config.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <fcntl.h>
#include <sys/mman.h>

TEST(test_load, load_model_config) {
    std::string model_path = "/home/yifanfang/MyKuiperLLama/tmp/test.bin";
    int32_t fd = open(model_path.data(), O_RDONLY); // open 成功返回非负整数文件描述符 (如 3, 4, 5...)，其中 0, 1, 2 是标准输入、输出、错误
    ASSERT_NE(fd, -1); // open 失败返回 -1

    FILE* file = fopen(model_path.data(), "rb");
    ASSERT_NE(file, nullptr);

    auto config = model::ModelConfig();
    ASSERT_EQ(fread(&config, sizeof(model::ModelConfig), 1, file), 1);
    ASSERT_EQ(config.dim, 16);
    ASSERT_EQ(config.head_num, 512);
    ASSERT_EQ(config.hidden_dim, 128);
    ASSERT_EQ(config.kv_head_num, 512);
    ASSERT_EQ(config.layer_num, 256);
    ASSERT_EQ(config.max_seq_len, 1024);
    ASSERT_EQ(config.vocab_size, 4);
}

TEST(test_load, test_model_weight) {
    std::string model_path = "/home/yifanfang/MyKuiperLLama/tmp/test.bin";
    int32_t fd = open(model_path.data(), O_RDONLY); // open 成功返回非负整数文件描述符 (如 3, 4, 5...)，其中 0, 1, 2 是标准输入、输出、错误
    ASSERT_NE(fd, -1); // open 失败返回 -1

    FILE* file = fopen(model_path.data(), "rb"); // fopen 成功返回非空 FILE* 指针
    ASSERT_NE(file, nullptr); // fopen 失败返回 NULL

    auto config = model::ModelConfig();
    ASSERT_EQ(fread(&config, sizeof(model::ModelConfig), 1, file), 1); // fread 成功返回实际读取的"元素个数"（不是字节数）

    ASSERT_EQ(fseek(file, 0, SEEK_END), 0); // fseek 将文件指针移动到文件末尾，成功返回 0
    size_t file_size = ftell(file); // ftell 获取当前文件指针位置（即文件总大小）
    
    struct stat st;
    ASSERT_NE(fstat(fd, &st), -1); // fstat 文件的状态信息，成功返回 0，失败返回 -1
    ASSERT_EQ(file_size, st.st_size);

    /*
    | ModelConfig  | 权重数据（浮点数数组） |
    | 大小为 sizeof(model::ModelConfig) | 剩余 dim × hidden_dim 部分全是浮点数 |
    */
    void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, MAP_FAILED);

    int32_t size = config.dim * config.hidden_dim;
    float* weight_data = reinterpret_cast<float*>(static_cast<char*>(data) + sizeof(model::ModelConfig));
    for (int32_t i = 0; i < size; i++) {
        ASSERT_EQ(*(weight_data + i), float(i));
    }

    std::vector<float> buffer(size); // std::vector 自动管理内存
    ASSERT_EQ(fseek(file, sizeof(model::ModelConfig), SEEK_SET), 0); // fseek 将文件指针移动到文件开头偏移 offset 位置，成功返回 0
    ASSERT_EQ(fread(buffer.data(), sizeof(float), size, file), size);
    for (int32_t i = 0; i < size; i++) {
        ASSERT_EQ(*(weight_data + i), buffer[i]);
    }
}

