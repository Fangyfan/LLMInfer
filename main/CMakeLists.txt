# ============================================
# 主可执行文件配置
# ============================================

message(STATUS "链接 QWEN3 支持库")

# 创建Qwen3模型推理程序
add_executable(qwen3_infer main_qwen3.cpp)

# 设置链接库搜索路径
target_link_directories(qwen3_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)

# 查找Qwen3所需依赖
find_package(absl REQUIRED)
find_package(re2 REQUIRED)
find_package(nlohmann_json REQUIRED)

# 链接库依赖
target_link_libraries(qwen3_infer 
    llm_infer  # 主库
    absl::base 
    re2::re2 
    nlohmann_json::nlohmann_json
)

# 设置CUDA可分离编译属性
set_target_properties(qwen3_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)