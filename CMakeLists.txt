# ============================================
# 基础配置部分
# ============================================

# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.16)

# 指定CUDA编译器路，必须在 project 之前设置
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# 项目定义
project(llama_infer CXX CUDA)

# 包含CUDA相关配置
include(cmake/cuda.cmake)

# 启用导出编译命令数据库（用于clangd等工具）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================
# 编译器配置
# ============================================

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 启用 O2 优化
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# ============================================
# 项目特性选项
# ============================================

# LLAMA3模型支持选项
option(LLAMA3_SUPPORT OFF)
if(LLAMA3_SUPPORT)
    message(STATUS "启用 LLAMA3 支持")
    add_definitions(-DLLAMA3_SUPPORT)
endif()

# QWEN2模型支持选项
option(QWEN2_SUPPORT OFF)
if(QWEN2_SUPPORT)
    message(STATUS "启用 QWEN2 支持")
    add_definitions(-DQWEN2_SUPPORT)
endif()

# QWEN3模型支持选项
option(QWEN3_SUPPORT OFF)
if(QWEN3_SUPPORT)
    message(STATUS "启用 QWEN3 支持")
    add_definitions(-DQWEN3_SUPPORT)
endif()

# ============================================
# 查找必需包
# ============================================

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

# ============================================
# 源代码收集
# ============================================

# Tensor相关源文件
aux_source_directory(code/source/tensor/ DIR_TENSOR)
# 基础模块源文件
aux_source_directory(code/source/base/ DIR_BASE)
# 操作符相关源文件
aux_source_directory(code/source/op/ DIR_OP)
# 模型相关源文件
aux_source_directory(code/source/model/ DIR_MODEL)
# CPU内核源文件
aux_source_directory(code/source/op/kernel/cpu DIR_KERNEL_CPU)
# CUDA内核源文件
aux_source_directory(code/source/op/kernel/cuda DIR_KERNEL_CUDA)
# 通用内核源文件
aux_source_directory(code/source/op/kernel/ DIR_KERNEL)
# 采样器源文件
aux_source_directory(code/source/sampler DIR_SAMPLE)

# ============================================
# 库文件输出目录
# ============================================

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ============================================
# 主库目标定义：构建llama动态库
# ============================================

# 创建共享库
add_library(llama SHARED 
    ${DIR_TENSOR}
    ${DIR_BASE}
    ${DIR_OP}
    ${DIR_KERNEL}
    ${DIR_MODEL}
    ${DIR_KERNEL_CPU}
    ${DIR_KERNEL_CUDA}
    ${DIR_SAMPLE}
)

# ============================================
# 链接库依赖
# ============================================

target_link_libraries(llama
    sentencepiece
    glog::glog
    gtest
    gtest_main
    pthread
    cudart
    armadillo
)

# 指定链接目录
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

# ============================================
# 包含目录配置
# ============================================

# 项目头文件目录
target_include_directories(llama PUBLIC 
    ${PROJECT_SOURCE_DIR}/code/include
)

# 第三方库头文件目录
target_include_directories(llama PUBLIC 
    ${glog_INCLUDE_DIR}
    ${Armadillo_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# CPM特定包含目录
if(USE_CPM)
    target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

# ============================================
# CUDA特定配置
# ============================================

# 启用CUDA分离编译（提升编译速度，支持大型CUDA项目）
set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================
# 子目录添加：构建测试和示例程序
# ============================================

# 测试代码目录
add_subdirectory(test)

# 演示代码目录
add_subdirectory(demo)

# ============================================
# 添加编译选项
# ============================================

# 解决Armadillo伪GCC警告
target_compile_definitions(llama PRIVATE
    ARMA_ALLOW_FAKE_GCC  # 允许Armadillo忽略伪GCC编译器检测
)