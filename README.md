# 基于现代 C++ 高性能大模型推理框架

[TOC]

## 项目介绍

- 基于现代 C++20 语法和 CUDA 并行编程，从零实现支持 Llama2 / Llama3.2 和 Qwen2.5 / Qwen3 的高性能大模型推理框架 。
- 采用 CPU 算子和 CUDA 算子双后端实现，同时支持 CUDA 加速和 Int8 量化 。
- 手动实现 Add、Embedding、RMSNorm（层归一化）、SwiGLU（激活函数）、RoPE（位置编码）、GEMV（矩阵向量乘法，支持 INT8 量化）、Softmax、Multi-Head Attention（GQA 多头注意力机制）、Argmax（采样）等 CPU / CUDA 大模型推理算子 。
- 使用 Warp 规约 / Block 规约 / 共享内存 / 向量化存取等技术优化 CUDA 算子性能 。
- 打通模型加载、Tokenizer、Prefill/Decode 推理链路，支持 KV Cache 与自回归文本生成流程 。
- 引入显存池（大/小块分级复用），减少频繁分配和释放开销，提升推理稳定性与吞吐率 。
- 通过统一内存分配器 / 设备抽象屏蔽 CPU / GPU 差异，通过算子注册 / 调度机制实现后端解耦与统一调用入口 。



## 第三方依赖

### LLama2 模型

#### Armadillo 数学库

用于高效实现大模型的 CPU 算子推理，安装步骤如下

##### 依赖库安装

MyKuiperLLama 依赖 Armadillo 数学库，而 Armadillo 需先安装底层依赖（OpenBLAS、LAPACK 等），命令如下

```bash
sudo apt install libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
```

无 root 权限，使用 conda 安装

```bash
conda create -n llm -c conda-forge openblas lapack arpack superlu
conda activate llm
```

##### 克隆源码

```bash
git clone https://gitee.com/mirrors/armadillo-code.git ~/armadillo
```

##### 编译安装（有 root 权限）

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

##### 编译安装到用户目录（无 root 权限）

```bash
# 进入源码目录
cd armadillo
# 创建 build 目录（分离编译文件与源码）
mkdir build && cd build
# 生成 Makefile（Release 模式，优化编译）
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/local \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DCMAKE_INSTALL_RPATH=$CONDA_PREFIX/lib \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      ..
# 编译：-j8 表示用 8 核，核数多可调大（如 -j16）
make -j$(nproc)
# 安装到用户目录
make install
# 安装测试
./tests1/smoke_test
```



#### Google Test 单元测试库

用于验证框架功能正确性，安装步骤如下

##### 克隆源码

```Shell
git clone git@github.com:google/googletest.git
```

##### 编译安装（有 root 权限）

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

##### 编译安装到用户目录（无 root 权限）

```bash
# 进入源码目录
cd googletest
# 创建 build 目录
mkdir build && cd build

# 生成动态库
# 临时取消 Anaconda 的 CPATH 干扰，仅对当前命令有效
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



#### Google Logging 日志库

用于框架运行日志记录，安装时需关闭不必要的编译选项，步骤如下

##### 克隆源码

```Shell
git clone git@github.com:google/glog.git
```

##### 编译安装（有 root 权限）

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

##### 编译安装到用户目录（无 root 权限）

```shell
git clone git@github.com:google/glog.git
# 进入源码目录
cd glog
# 切换到 v0.7.1 标签
git checkout v0.7.1
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile：关闭 WITH_GFLAGS 和 WITH_GTEST（避免额外依赖）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DWITH_GFLAGS=OFF \
  -DWITH_GTEST=OFF \
  ..
# 编译
make -j$(nproc)
# 安装
make install
```



#### SentencePiece 分词库

用于大模型输入文本的分词处理，安装步骤如下

##### 克隆源码

```Shell
git clone git@github.com:google/sentencepiece.git
```

##### 编译安装（有 root 权限）

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

##### 编译安装到用户目录（无 root 权限）

```bash
# 进入源码目录
cd sentencepiece
# 创建 build 目录
mkdir build && cd build
# 生成 Makefile（Release 模式）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  ..
# 编译
make -j$(nproc)
# 安装
make install
```



### LLama3.2 模型





## 项目编译方法

```bash
git clone git@github.com:Fangyfan/MyKuiperLLama.git
cd MyKuiperLLama
```

```bash
mkdir build
cd build
# 需要安装上述的第三方依赖
cmake ..
make -j$(nproc)
```



## GTest 单元测试

在编译后的 build 目录下，执行以下命令

```bash
cd test
./test_llm
```



## 分词器与模型权重下载

下载 TinyLLama 的模型和分词器到 ~/MyKuiperLLama/models

1. SentencePiece (SPE) 分词器：https://huggingface.co/yahma/llama-7b-hf/blob/main/tokenizer.model
2. LLama2 非量化模型权重：https://huggingface.co/karpathy/tinyllamas/blob/main/stories110M.bin
3. LLama2 INT8 量化模型权重：https://huggingface.co/fushenshen/lession_model/blob/main/chat_q8.bin



## LLama2 模型

参考资料：

https://zhuanlan.zhihu.com/p/649756898

https://www.armcvai.cn/2024-10-21/llama1-3-model.html

### CPU 推理

在 demo/main.cpp 中修改以下代码，以支持 CPU 非量化推理

```cpp
bool is_quant_model = false;
base::Status status = model.init(base::DeviceType::DeviceCPU);
```

#### 运行命令

在编译后的 build 目录下，执行以下命令

```bash
cd demo
./llama_infer /home/yifanfang/MyKuiperLLama/models/stories110M.bin /home/yifanfang/MyKuiperLLama/models/tokenizer.model
```

#### 运行结果

在一张 NVIDIA GeForce RTX 4090 上的运行结果

```bash
Llama2 model generating...
hello, who was a little girl. She was three years old and loved to explore. One day, she was walking in the park when she saw a big, red balloon. She was so excited and ran over to it. She reached out to grab it, but it was too high.
Suddenly, a man appeared. He was wearing a big hat and had a big smile on his face. He said, "Hello there! Would you like to have a balloon?"
The little girl nodded her head and the man reached up and grabbed the balloon. He handed
steps: 128
time(s): 166.207
steps/s: 0.770125
```



### CUDA 非量化推理

在 demo/main.cpp 中修改以下代码，以支持 CUDA 非量化推理

```cpp
bool is_quant_model = false;
base::Status status = model.init(base::DeviceType::DeviceCUDA);
```

#### 运行命令

在编译后的 build 目录下，执行以下命令

```bash
cd demo
./llama_infer /home/yifanfang/MyKuiperLLama/models/stories110M.bin /home/yifanfang/MyKuiperLLama/models/tokenizer.model
```

#### 运行结果

在一张 NVIDIA GeForce RTX 4090 上的运行结果

```bash
Llama2 model generating...
hello, who was a little girl. She was three years old and loved to explore. One day, she was walking in the park when she saw a big, red balloon. She was so excited and ran over to it. She reached out to grab it, but it was too high.
Suddenly, a man appeared. He was wearing a big hat and had a big smile on his face. He said, "Hello there! Would you like to have a balloon?"
The little girl nodded her head and the man reached up and grabbed the balloon. He handed
steps: 128
time(s): 0.110267
steps/s: 1160.82
```



### CUDA INT8 量化推理

在 demo/main.cpp 中修改以下代码，以支持 CUDA INT8 量化推理

```cpp
bool is_quant_model = true;
base::Status status = model.init(base::DeviceType::DeviceCUDA);
```

#### 运行命令

在编译后的 build 目录下，执行以下命令

```bash
cd demo
./llama_infer /home/yifanfang/MyKuiperLLama/models/chat_q8.bin /home/yifanfang/MyKuiperLLama/models/tokenizer.model
```

#### 运行结果

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



## Qwen2.5 模型



## Qwen3 模型
