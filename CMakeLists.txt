# ============================================
# 基础配置部分
# ============================================

# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.16)

# 指定CUDA编译器路，必须在 project 之前设置
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# 项目定义
project(llm_infer CXX CUDA)

# 默认 Release 构建，避免无优化编译
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 把 CUDA 架构固定成 RTX 4090 的 Ada Lovelace sm89
set(CMAKE_CUDA_ARCHITECTURES 89)

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
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================
# 项目特性选项
# ============================================

# QWEN3模型支持选项
message(STATUS "启用 QWEN3 支持")

# ============================================
# 查找必需包
# ============================================

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ============================================
# 源代码收集
# ============================================

aux_source_directory(code/source/tensor/ DIR_TENSOR)
aux_source_directory(code/source/base/ DIR_BASE)
aux_source_directory(code/source/op/ DIR_OP)
aux_source_directory(code/source/model/ DIR_MODEL)
aux_source_directory(code/source/op/kernel/cpu DIR_KERNEL_CPU)
aux_source_directory(code/source/op/kernel/cuda DIR_KERNEL_CUDA)
aux_source_directory(code/source/op/kernel/ DIR_KERNEL)
aux_source_directory(code/source/sampler DIR_SAMPLE)

# ============================================
# 库文件输出目录
# ============================================

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ============================================
# 主库目标定义：构建infer动态库
# ============================================

# 创建共享库
add_library(llm_infer SHARED 
    ${DIR_TENSOR}
    ${DIR_BASE}
    ${DIR_OP}
    ${DIR_KERNEL}
    ${DIR_MODEL}
    ${DIR_KERNEL_CPU}
    ${DIR_KERNEL_CUDA}
    ${DIR_SAMPLE}
)

target_compile_options(llm_infer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -DNDEBUG
        -march=native
        -mtune=native
        -fno-math-errno
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -DNDEBUG
        --extra-device-vectorization
        -Xptxas=-O3
    >
)

# 可选：先不要默认打开，确认数值无问题后再启用
option(LLM_USE_CUDA_FAST_MATH "Enable CUDA fast math for inference kernels" OFF)

if(LLM_USE_CUDA_FAST_MATH)
    target_compile_options(llm_infer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

# ============================================
# 链接库依赖
# ============================================

target_link_libraries(llm_infer
    sentencepiece
    glog::glog
    gtest
    gtest_main
    pthread
    armadillo
    CUDA::cudart
)

# 指定链接目录
target_link_directories(llm_infer PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

# ============================================
# 包含目录配置
# ============================================

# 项目头文件目录
target_include_directories(llm_infer PUBLIC 
    ${PROJECT_SOURCE_DIR}/code/include
)

# 第三方库头文件目录
target_include_directories(llm_infer PUBLIC 
    ${glog_INCLUDE_DIR}
    ${Armadillo_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# CPM特定包含目录
if(USE_CPM)
    target_include_directories(llm_infer PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

# ============================================
# CUDA特定配置
# ============================================

# 启用CUDA分离编译（提升编译速度，支持大型CUDA项目）
set_target_properties(llm_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================
# 子目录添加：构建测试和示例程序
# ============================================

# 测试代码目录
# add_subdirectory(test)

# 演示代码目录
add_subdirectory(main)

# ============================================
# 添加编译选项
# ============================================

# 解决Armadillo伪GCC警告
target_compile_definitions(llm_infer PRIVATE
    ARMA_ALLOW_FAKE_GCC  # 允许Armadillo忽略伪GCC编译器检测
)