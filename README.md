# 基于现代 C++ 高性能大模型推理框架

[TOC]

## 项目介绍

- 基于现代 C++17 语法和 CUDA 并行编程，从零实现支持 Llama2/3 和 Qwen3 的高性能大模型推理框架 。
- 采用 CPU 算子和 CUDA 算子双后端实现，同时支持 CUDA 加速和 Int8 量化 。
- 手动实现 Add、Embedding、RMSNorm（层归一化）、SwiGLU（激活函数）、RoPE（位置编码）、GEMV（矩阵向量乘法，支持 INT8 量化）、Softmax、Multi-Head Attention（GQA 多头注意力机制）、Argmax（采样）等 CPU / CUDA 大模型推理算子 。
- 使用 Warp 规约 / Block 规约 / 共享内存 / 向量化存取等技术优化 CUDA 算子性能 。
- 打通模型加载、Tokenizer、Prefill、Decode 推理链路，支持 KV Cache 与自回归文本生成流程 。
- 引入显存池（大/小块分级复用），减少频繁分配和释放开销，提升推理稳定性与吞吐率 。
- 通过统一内存分配器 / 设备抽象屏蔽 CPU / GPU 差异，通过算子注册 / 调度机制实现后端解耦与统一调用入口 。



## LLama2 第三方依赖

### Armadillo 数学库

用于高效实现大模型的 CPU 算子推理，安装步骤如下

依赖库安装

LLMInfer 依赖 Armadillo 数学库，而 Armadillo 需先安装底层依赖（OpenBLAS、LAPACK 等），命令如下

```bash
sudo apt install libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
```

无 root 权限，使用 conda 安装，或者克隆源码编译安装

```bash
conda create -n llm -c conda-forge openblas lapack arpack superlu
conda activate llm
```

克隆源码

https://gitee.com/mirrors/armadillo-code

```bash
git clone https://gitee.com/mirrors/armadillo-code.git ~/armadillo
```

编译安装（有 root 权限）

```bash
# 进入源码目录
cd armadillo
# 创建 build 目录（分离编译文件与源码）
mkdir build && cd build
# 生成 Makefile（Release 模式，优化编译）
cmake -DCMAKE_BUILD_TYPE=Release ..
# 编译：-j8 表示用 8 核，核数多可调大（如 -j16），$(nproc) 表示当前系统的核数
make -j$(nproc)
# 安装到系统目录（需 sudo 权限）
sudo make install
# 安装测试
./tests1/smoke_test
```

编译安装到用户目录（无 root 权限）

```bash
# 进入源码目录
cd armadillo
# 创建 build 目录（分离编译文件与源码）
mkdir build && cd build
# 生成 Makefile（Release 模式，优化编译）
# -DBUILD_SHARED_LIBS=ON 开启动态库编译（默认OFF，只生成静态库）
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/local \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DCMAKE_INSTALL_RPATH=$CONDA_PREFIX/lib \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DBUILD_SHARED_LIBS=ON \
      ..
# 编译：-j8 表示用 8 核，核数多可调大（如 -j16）
make -j$(nproc)
# 安装到用户目录
make install
# 安装测试
./tests1/smoke_test
```



### Google Test 单元测试库

用于验证框架功能正确性，安装步骤如下

克隆源码

https://github.com/google/googletest

```Shell
git clone git@github.com:google/googletest.git
```

编译安装（有 root 权限）

```Shell
# 进入源码目录
cd googletest
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile（Release 模式）
cmake -DCMAKE_BUILD_TYPE=Release .. 
# 编译 
make -j$(nproc)
# 安装（默认路径：头文件 /usr/local/include，库文件 /usr/local/lib）
sudo make install
```

编译安装到用户目录（无 root 权限）

```bash
# 进入源码目录
cd googletest
# 创建 build 目录
mkdir build && cd build

# 生成动态库
# 临时取消 Anaconda 的 CPATH 干扰，仅对当前命令有效
# -DBUILD_SHARED_LIBS=ON 开启动态库编译（默认OFF，只生成静态库）
unset CPATH && \
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DBUILD_GMOCK=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CXX_FLAGS="-I$HOME/googletest/googletest/include" \
  -DCMAKE_C_FLAGS="-I$HOME/googletest/googletest/include"

# 多线程编译（速度更快）
make -j$(nproc)
# 安装到用户目录
make install
```



### Google Logging 日志库

用于框架运行日志记录，安装时需关闭不必要的编译选项，步骤如下

克隆源码

https://github.com/google/glog

```Shell
git clone git@github.com:google/glog.git
```

编译安装（有 root 权限）

```Shell
# 进入源码目录
cd glog
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile：关闭 WITH_GFLAGS 和 WITH_GTEST（关闭 GFLAGS 和 GTEST 依赖，避免额外依赖）
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF ..
# 编译
make -j$(nproc)
# 安装
sudo make install
```

编译安装到用户目录（无 root 权限）

```shell
git clone git@github.com:google/glog.git
# 进入源码目录
cd glog
# 切换到 v0.7.1 标签
git checkout v0.7.1
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile：关闭 WITH_GFLAGS 和 WITH_GTEST（避免额外依赖）
# -DBUILD_SHARED_LIBS=ON 开启动态库编译（默认OFF，只生成静态库）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DWITH_GFLAGS=OFF \
  -DWITH_GTEST=OFF \
  -DBUILD_SHARED_LIBS=ON \
  ..
# 编译
make -j$(nproc)
# 安装
make install
```



### SentencePiece 分词库

用于大模型输入文本的分词处理，安装步骤如下

克隆源码

https://github.com/google/sentencepiece

```Shell
git clone git@github.com:google/sentencepiece.git
```

编译安装（有 root 权限）

```Shell
# 进入源码目录
cd sentencepiece
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile（Release 模式）
cmake -DCMAKE_BUILD_TYPE=Release .. 
# 编译
make -j$(nproc)
# 安装
sudo make install
```

编译安装到用户目录（无 root 权限）

```bash
# 进入源码目录
cd sentencepiece
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile（Release 模式）
# -DBUILD_SHARED_LIBS=ON 开启动态库编译（默认OFF，只生成静态库）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DBUILD_SHARED_LIBS=ON \
  ..
# 编译
make -j$(nproc)
# 安装
make install
```



## LLama3.2 / Qwen3 第三方依赖

### Abseil (absl) C++ 标准库扩展

克隆源码

https://github.com/abseil/abseil-cpp

```bash
git clone git@github.com:abseil/abseil-cpp.git
```

编译安装到用户目录（无 root 权限）

```bash
# 进入源码目录
cd abseil-cpp
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile（Release 模式）
# -DBUILD_SHARED_LIBS=ON 开启动态库编译（默认OFF，只生成静态库）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DBUILD_SHARED_LIBS=ON \
  ..
# 编译
make -j$(nproc)
# 安装
make install
```



### re2 正则表达式库

克隆源码

https://github.com/google/re2

```bash
git clone git@github.com:google/re2.git
```

编译安装到用户目录（无 root 权限）

```bash
# 进入源码目录
cd re2
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile（Release 模式）
# -DBUILD_SHARED_LIBS=ON 开启动态库编译（默认OFF，只生成静态库）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DBUILD_SHARED_LIBS=ON \
  ..
# 编译
make -j$(nproc)
# 安装
make install
```



### nlohmann/json 读取 JSON 库

克隆源码

https://github.com/nlohmann/json

```bash
git clone git@github.com:nlohmann/json.git
```

编译安装到用户目录（无 root 权限）

```bash
# 进入源码目录
cd json
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile（Release 模式）
# -DBUILD_SHARED_LIBS=ON 开启动态库编译（默认OFF，只生成静态库）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DBUILD_SHARED_LIBS=ON \
  ..
# 编译
make -j$(nproc)
# 安装
make install
```



## 环境变量设置

由于使用 conda 安装了相关依赖，但是 conda 又引入了一些库，其中有些 conda 库与系统目录下的库发生冲突：比如我希望编译时库和链接时库是相同的，但是很多时候一个是系统库，另一个是 conda 库，导致版本冲突，编译失败 。

我们希望项目在编译和链接时，优先在系统目录中查找库，如果系统目录找不到，再去查找 conda 库 。

因此，在 ~/.bashrc 文件中添加

```bash
# prefer local install
export HOME_LOCAL="$HOME/local"

# ALWAYS prefer local include/lib before system ones
export LD_LIBRARY_PATH="${HOME_LOCAL}/lib:/usr/local/lib:/usr/lib"
export LIBRARY_PATH="${HOME_LOCAL}/lib:/usr/local/lib:/usr/lib"

# Put local include first; do NOT add conda include to CPATH/CXX_INCLUDE_PATH
export CPATH="${HOME_LOCAL}/include:/usr/local/include:/usr/include"
export C_INCLUDE_PATH="${HOME_LOCAL}/include:/usr/local/include:/usr/include"
export CXX_INCLUDE_PATH="${HOME_LOCAL}/include:/usr/local/include:/usr/include"

# Only if conda is active, append its lib path to LD_LIBRARY_PATH, but DO NOT prepend conda include
if [ -n "${CONDA_PREFIX}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
    export PKG_CONFIG_PATH="${HOME_LOCAL}/lib/pkgconfig:${CONDA_PREFIX}/lib/pkgconfig:/usr/local/lib/pkgconfig:/usr/lib/pkgconfig"
else
    export PKG_CONFIG_PATH="${HOME_LOCAL}/lib/pkgconfig:/usr/local/lib/pkgconfig:/usr/lib/pkgconfig"
fi

# Ensure cmake prefers ${HOME_LOCAL} for find_package lookups
export CMAKE_PREFIX_PATH="${HOME_LOCAL}:${CMAKE_PREFIX_PATH}"
```

运行如下命令，使环境变量生效

```bash
source ~/.bashrc
```



## 项目编译方法

克隆源码

```bash
git clone git@github.com:Fangyfan/LLMInfer.git
# 进入源码目录
cd LLMInfer
```

编译

```bash
# 创建 build 目录
mkdir build && cd build
# 清空 build 目录
rm -rf *
# 需要安装上述的第三方依赖
cmake ..
# 编译
make -j$(nproc)
```



## VS Code 断点调试

用 Debug 模式进行编译

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

修改 launch.json 文件

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) launch",
            "type": "cppdbg",
            "request": "launch",
            
            "program": "${workspaceFolder}/build/demo/llama_infer",
            
            "args": [
                "${workspaceFolder}/models/llama2/stories110M.bin",
                "${workspaceFolder}/models/llama2/tokenizer.model",
            ],

            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb (为 gdb 启用整齐打印)",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description": "Set Disassembly Flavor to Intel",
                    "text": "set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ],
        }
    ]
}
```

在程序中打上断点，点击左侧 “运行和调试” 图标，在左侧栏上面点击绿色三角符号（开始调试）即可



## GTest 单元测试

在编译后的 build 目录下，执行以下命令

```bash
cd test
./test_llm
```



## 分词器与模型权重下载

下载 TinyLLama 的模型和分词器到 ~/LLMInfer/models

1. SentencePiece (SPE) 分词器：https://huggingface.co/yahma/llama-7b-hf/blob/main/tokenizer.model
2. LLama2 非量化模型权重：https://huggingface.co/karpathy/tinyllamas/blob/main/stories110M.bin
3. LLama2 INT8 量化模型权重：https://huggingface.co/fushenshen/lession_model/blob/main/chat_q8.bin
3. LLama3.2 非量化模型权重：https://huggingface.co/fushenshen/lession_model/tree/main/llama32_1bnq.bin
3. LLama3.2 BPE 分词器：https://huggingface.co/fushenshen/lession_model/tree/main/tokenizer.json
3. Qwen3 模型权重和分词器：https://huggingface.co/fushenshen/lession_model/blob/main/qwen3-0.6b.zip



## LLama2 模型

参考资料：

https://zhuanlan.zhihu.com/p/649756898

https://www.armcvai.cn/2024-10-21/llama1-3-model.html

### CPU 推理

在 demo/main.cpp 中修改以下代码，以支持 CPU 非量化推理

```cpp
bool is_quant_model = false;
model::Llama2Model model(base::TokenizerType::EncodeSpe, ...);
base::Status status = model.init(base::DeviceType::DeviceCPU);
```

在编译后的 build 目录下，执行以下命令

```bash
cd demo
./llama_infer /home/yifanfang/LLMInfer/models/llama2/stories110M.bin /home/yifanfang/LLMInfer/models/llama2/tokenizer.model
```

在一张 NVIDIA GeForce RTX 4090 上的运行结果

```bash
Llama2 model generating...
hello, who was a little girl. She was three years old and loved to explore. One day, she was walking in the park when she saw a big, red balloon. She was so excited and ran over to it. She reached out to grab it, but it was too high.
Suddenly, a man appeared. He was wearing a big hat and had a big smile on his face. He said, "Hello there! Would you like to have a balloon?"
The little girl nodded her head and the man reached up and grabbed the balloon. He handed
steps: 128
time(s): 5.62209
steps/s: 22.7673
```



### CUDA 非量化推理

在 demo/main.cpp 中修改以下代码，以支持 CUDA 非量化推理

```cpp
bool is_quant_model = false;
model::Llama2Model model(base::TokenizerType::EncodeSpe, ...);
base::Status status = model.init(base::DeviceType::DeviceCUDA);
```

在编译后的 build 目录下，执行以下命令

```bash
cd demo
./llama_infer /home/yifanfang/LLMInfer/models/llama2/stories110M.bin /home/yifanfang/LLMInfer/models/llama2/tokenizer.model
```

在一张 NVIDIA GeForce RTX 4090 上的运行结果

```bash
Llama2 model generating...
hello, who was a little girl. She was three years old and loved to explore. One day, she was walking in the park when she saw a big, red balloon. She was so excited and ran over to it. She reached out to grab it, but it was too high.
Suddenly, a man appeared. He was wearing a big hat and had a big smile on his face. He said, "Hello there! Would you like to have a balloon?"
The little girl nodded her head and the man reached up and grabbed the balloon. He handed
steps: 128
time(s): 0.109198
steps/s: 1172.18
```



### CUDA INT8 量化推理

在 demo/main.cpp 中修改以下代码，以支持 CUDA INT8 量化推理

```cpp
bool is_quant_model = true;
model::Llama2Model model(base::TokenizerType::EncodeSpe, ...);
base::Status status = model.init(base::DeviceType::DeviceCUDA);
```

在编译后的 build 目录下，执行以下命令

```bash
cd demo
./llama_infer /home/yifanfang/LLMInfer/models/llama2/chat_q8.bin /home/yifanfang/LLMInfer/models/llama2/tokenizer.model
```

在一张 NVIDIA GeForce RTX 4090 上的运行结果

```bash
Llama2-quant8 model generating...
hello, and then I'll be back.

JASON: (smiling) Hey, I'm glad you're here.

JASON'S MOTHER: (smiling) You're welcome.

JASON: (smiling) I'm glad I could help.

JASON'S FATHER: (smiling) You're a good son.

JASON: (smiling) Thanks, Dad.

JASON'S MOTHER: (smiling) You're welcome.


steps: 128
time(s): 0.362295
steps/s: 353.303
```



## LLama3.2 模型

### CUDA 非量化推理

在 build 目录下重新编译

```bash
cmake .. -DLLAMA3_SUPPORT=ON
make -j$(nproc)
```

在 demo/main.cpp 中修改以下代码，以支持 CUDA 非量化推理

```cpp
bool is_quant_model = false;
model::Llama2Model model(base::TokenizerType::EncodeBpe, ...);
base::Status status = model.init(base::DeviceType::DeviceCUDA);
```

在编译后的 build 目录下，执行以下命令

```bash
cd demo
./llama_infer /home/yifanfang/LLMInfer/models/llama3.2/llama32_1bnq.bin /home/yifanfang/LLMInfer/models/llama3.2/tokenizer.json
```

在一张 NVIDIA GeForce RTX 4090 上的运行结果

```bash
Llama3.2 model generating...
hello, i am a little bit confused with the same thing about the same time, ...
steps: 128
time(s): 0.715422
steps/s: 178.915
```



## Qwen3 模型

### CUDA 非量化推理

在 build 目录下重新编译

```bash
cmake .. -DQWEN3_SUPPORT=ON
make -j$(nproc)
```

在 demo/main_qwen3.cpp 中修改以下代码，以支持 CUDA 非量化推理

```cpp
bool is_quant_model = false;
model::Qwen3Model model(base::TokenizerType::EncodeBpe, ...);
base::Status status = model.init(base::DeviceType::DeviceCUDA);
```

在编译后的 build 目录下，执行以下命令

```bash
cd demo
./qwen3_infer /home/yifanfang/LLMInfer/models/qwen3/qwen0.6.bin2 /home/yifanfang/LLMInfer/models/qwen3/tokenizer.json
```

在一张 NVIDIA GeForce RTX 4090 上的运行结果

```bash
What is AI?
Qwen3 model generating...
<think>
Okay, the user is asking what AI is. Let me start by explaining the basic definition. AI refers to artificial intelligence, which is a technology that allows machines to perform tasks that typically require human intelligence. I should mention key areas like problem-solving, learning, and decision-making.

I need to make sure the explanation is clear and covers the main points. Also, I should include examples to help illustrate the concept. Maybe mention applications in different fields like healthcare, finance, and education. It's important to highlight that AI is a subset of computer science and has various uses.

Wait, should I add anything about the difference between AI and machine learning? That might be useful. Also, maybe touch on the ethical considerations, but since the user didn't ask about that, perhaps keep it simple. Let me check if there's any technical jargon I should avoid. Keep the language straightforward and accessible.
</think>

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. It involves creating systems that can perform tasks typically requiring human intelligence, such as learning, problem-solving, reasoning, and decision-making. AI systems can be designed to:

1. **Learn from data**: Using algorithms to analyze patterns and improve over time.
2. **Make decisions**: Solving problems or making choices based on data.
3. **Interact with humans**: Understanding natural language, recognizing patterns, and adapting to user inputs.

AI is a subset of computer science and has applications in various fields, including healthcare, finance, education, and autonomous systems. It's continuously evolving, with advancements in machine learning and deep learning enabling more complex AI capabilities.
steps: 344
time(s): 1.27214
steps/s: 270.41
```

