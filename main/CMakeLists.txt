# ============================================
# 主可执行文件配置
# ============================================

message(STATUS "链接 QWEN3 支持库")

# 查找Qwen3所需依赖
find_package(absl REQUIRED)
find_package(re2 REQUIRED)
find_package(nlohmann_json REQUIRED)

# ============================================
# Qwen3 BF16 推理程序
# ============================================

add_executable(qwen3_infer main_qwen3.cpp)

target_link_directories(qwen3_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(qwen3_infer 
    llm_infer
    absl::base 
    re2::re2 
    nlohmann_json::nlohmann_json
)

set_target_properties(qwen3_infer PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================
# Qwen3 AWQ INT4 推理程序
# ============================================

add_executable(qwen3_awq_infer main_qwen3_awq.cpp)

target_link_directories(qwen3_awq_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(qwen3_awq_infer 
    llm_infer
    absl::base 
    re2::re2 
    nlohmann_json::nlohmann_json
)

set_target_properties(qwen3_awq_infer PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)