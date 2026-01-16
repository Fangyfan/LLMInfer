#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    // 初始化 Google Test 框架 (必须在使用任何 Google Test 功能之前调用，否则测试框架无法正常工作)
    testing::InitGoogleTest(&argc, argv);
    // 初始化 glog 日志系统，指定日志文件的前缀名（如 Kuiper.INFO, Kuiper.ERROR），帮助区分不同应用程序的日志文件
    google::InitGoogleLogging("Kuiper");
    // glog 的全局标志变量，设置日志文件的输出目录为 ./log/ ，程序需要有该目录的写入权限
    // FLAGS_log_dir = "./log/";
    // 在将日志写入文件的同时，也输出到标准错误(stderr)，开发调试时方便查看控制台输出，生产环境通常设为 false，避免影响性能
    FLAGS_alsologtostderr = true;
    
    // 确保在单卡上面分配显存，当前正在使用的 GPU 设备 ID
    CHECK(cudaSetDevice(0) == cudaSuccess);

    // 记录 INFO 级别的日志，输出程序开始执行的日志信息
    LOG(INFO) << "Start Test..." << std::endl;
    // 执行所有 TEST() 宏定义的测试用例，返回 0 所有测试通过，返回 1 有测试失败
    return RUN_ALL_TESTS();
}