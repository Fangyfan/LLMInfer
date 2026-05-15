# ============================================
# 包含上级目录配置
# ============================================

# 包含CUDA相关配置
include(../cmake/cuda.cmake)

# ============================================
# 查找外部依赖包
# ============================================

# 查找Google Test框架
find_package(GTest REQUIRED)

# 查找Google Logging库
find_package(glog REQUIRED)

# ============================================
# 外部链接库变量定义
# ============================================

set(link_ext_lib 
    glog::glog 
    GTest::gtest
)

# ============================================
# 收集测试源代码
# ============================================

# 收集所有测试文件
aux_source_directory(../test DIR_TEST)
# 收集操作符相关测试文件
aux_source_directory(../test/test_op DIR_TEST_OP)
# 收集张量相关测试文件
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)

# ============================================
# 创建测试可执行文件
# ============================================

add_executable(test 
    ${DIR_TEST}          # 通用测试
    ${DIR_TEST_OP}       # 算子测试
    ${DIR_TEST_TENSOR}   # 张量测试
)

# ============================================
# 链接外部库依赖
# ============================================

target_link_libraries(test ${link_ext_lib})

# ============================================
# 设置包含目录
# ============================================

# Google Logging头文件目录
target_include_directories(test PUBLIC ${glog_INCLUDE_DIR})

# Google Test头文件目录
target_include_directories(test PUBLIC ${GTest_INCLUDE_DIR})

# 项目主头文件目录
target_include_directories(test PUBLIC ../code/include)

# ============================================
# 设置链接目录
# ============================================

target_link_directories(test PUBLIC ${PROJECT_SOURCE_DIR}/lib)

# ============================================
# 模型特定依赖配置
# ============================================

message(STATUS "链接 QWEN3 模型特定依赖")

# 查找模型特定依赖包
find_package(absl REQUIRED)
find_package(re2 REQUIRED)
find_package(nlohmann_json REQUIRED)

# 链接依赖到主库（注意：这里链接到llm_infer库，不是test）
target_link_libraries(llm_infer 
    absl::base 
    re2::re2 
    nlohmann_json::nlohmann_json
)

# ============================================
# 链接主项目库
# ============================================

target_link_libraries(test llm_infer)

# ============================================
# 目标属性设置
# ============================================

# 设置工作目录为项目根目录（便于访问测试数据）
set_target_properties(test PROPERTIES 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# 启用CUDA可分离编译
set_target_properties(test PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)