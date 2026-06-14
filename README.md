# 基于现代 C++ 高性能大模型推理框架



## 项目介绍

基于现代 C++ 语言实现的高性能大语言模型（LLM）推理引擎，手写 CUDA 算子，支持 BF16 与 INT4 (AWQ) 量化推理，**单 Batch 请求 Decode 阶段 TPOT 超越 vLLM** 。

项目实现特点如下：

- **现代 C++ 17 标准实现**（STL、指针、智能指针、封装、继承、多态、模板、类型转换、RAII 内存管理）
- NVIDIA Ada Lovelace GPU 架构（RTX 4090、SM89）**CUDA 算子实现与优化** ：ResidualAdd、Embedding、RMSNorm、SwiGLU、RoPE、GEMV、Softmax、Grouped-Query Attention (GQA)、Argmax Sampling 等
- 支持 **Qwen3 模型全链路 FP32 / BF16 推理**，其中 BF16 类型在 C++17 中以 2 字节数据格式 `std::uint16_t` 存储
- 支持 **CPU / CUDA 双后端** 算子实现，通过统一内存分配器 / 设备抽象屏蔽 CPU / GPU 差异，通过算子注册 / 调度机制实现后端解耦与统一调用入口
- 打通完整的端到端推理流水线（**权重加载 → Tokenizer → Embedding → Prefill → Decode**）
- 支持 **KV Cache** 与自回归文本生成流程，缓存 KV 加快推理速度
- 引入 **CUDA 显存池**（大 / 小块分级复用 +  Best-Fit 分配策略），减少频繁分配和释放开销
- 通过 **mmap 内存映射** 实现 **零拷贝融合算子权重加载** 
- 实现 Qwen3 模型 Decode 阶段 GQA 实现 **短上下文 FlashAttention** 和 **长上下文 KV-split FlashDecoding** 的 dispatch 策略以减少 HBM 访存并且提高 GPU 并行度，端到端性能提升 41.4%
- 实现 Qwen3 模型 **BF16 / FP32 混合精度融合算子**（QKV Packed GEMV、QK-Norm + QK-RoPE、GEMV + ResidualAdd、Gate-Up Packed GEMV + SwiGLU、ResidualAdd + RMSNorm），通过 **FP32 乘加减少精度损失**，采用 **bf16x8 向量化访存**，Warp Shuffle 优化 Block 规约，端到端性能提升 27.7%
- 手写 BF16 / FP32 混合精度 GEMV 融合算子 **较 cuBLASLt 实现 1.68x 加速** 
- 支持 **AWQ INT4 量化推理 (W4A16)**，量化融合算子权重 mmap 零拷贝加载，采用 AutoAWQ N-packed 量化格式
- 实现 Qwen3-4B-AWQ 模型 **INT4 / BF16 / FP32 混合精度 GEMV 融合算子**（INT4 QKV Packed GEMV、INT4 GEMV + ResidualAdd、INT4 Gate-Up Packed GEMV + SwiGLU）
- **Qwen3-4B BF16 单 Batch Decode 达到理论极限性能的 89%，超越 vLLM 性能，较 PyTorch 实现 1.87x 加速** 
- Qwen3-4B-AWQ INT4 量化模型推理较 Qwen3-4B 纯 BF16 推理实现 2.11x 端到端加速



## Qwen3-4B (BF16) 性能对比

|                 | LLMInfer     | vLLM (CUDA Graph) | vLLM (Eager) | PyTorch       | 理论极限     |
| --------------- | ------------ | ----------------- | ------------ | ------------- | ------------ |
| **Decode TPOT** | 8.95ms/token | 9.15ms/token      | 9.65ms/token | 16.84ms/token | 7.98ms/token |



## Qwen3-4B 模型推理（BF16）

在 build 目录下编译

```bash
cd LLMInfer
mkdir build && cd build
rm -rf *
cmake ..
make -j$(nproc)
```

在编译后的 build 目录下，执行以下命令

```bash
cd main
./qwen3_infer /home/yifanfang/LLMInfer/models/qwen3/qwen3_4b_bf16.bin /home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B/tokenizer.json
```

在一张 NVIDIA GeForce RTX 4090 上的运行结果

```bash
What is AI?
Qwen3 model warmup...
Qwen3 model generating...
<think>
Okay, the user asked, "What is AI?" I need to explain AI in a clear and concise way. Let me start by defining AI, maybe mention that it's a branch of computer science. Then, I should explain the key characteristics, like the ability to perform tasks that typically require human intelligence. I should list examples of tasks, such as problem-solving, learning, reasoning, perception, and language understanding.

Wait, I should also differentiate between narrow AI and general AI. Narrow AI is designed for specific tasks, like voice assistants or recommendation systems. General AI is still theoretical and refers to systems that can handle any intellectual task a human can. It's important to note that current AI is mostly narrow.

I should also mention the applications of AI in various fields, like healthcare, finance, transportation, etc. Maybe include some examples like machine learning, natural language processing, computer vision. Also, touch on the challenges and ethical considerations, such as bias, privacy, and job displacement.

But the user might be looking for a basic definition without too much technical jargon. So I should keep it simple. Let me structure the answer with a definition, key features, examples, and applications. Make sure to highlight that AI is not a single technology but a broad field with many subfields. Also, clarify that AI systems are not conscious or self-aware, which is a common misconception.

I need to check if I'm missing any important points. Maybe mention that AI is developed through algorithms and data, and that it's used to automate tasks and make decisions. Also, the difference between AI and machine learning. Oh right, machine learning is a subset of AI. So I should explain that as well.

Wait, the user might not know the difference between AI and machine learning. So I should clarify that AI is the broader concept, and machine learning is a part of it. Also, mention other subfields like robotics, expert systems, etc.

I should also mention the history of AI, like the Dartmouth Conference in 1956, but maybe that's too detailed. The user might not need that. Focus on the current understanding and applications.

Make sure the answer is easy to understand, not too technical. Avoid using complex terms unless necessary. Use examples to illustrate points. Conclude by summarizing the key points and maybe mention the future of AI.
</think>

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines or software. It involves creating systems that can perform tasks typically requiring human intelligence, such as problem-solving, learning, reasoning, perception, and language understanding. AI is a broad field that encompasses various subfields, including machine learning, natural language processing, computer vision, robotics, and expert systems.

### Key Characteristics of AI:
1. **Problem-Solving**: AI systems can analyze complex problems and devise solutions, often more efficiently than humans.
2. **Learning**: Machine learning algorithms improve their performance through experience, using data to identify patterns and make predictions.
3. **Reasoning**: AI can perform logical deductions, infer conclusions, and make decisions based on data.
4. **Perception**: Systems can interpret sensory input (e.g., images, sounds) to recognize objects, faces, or speech.
5. **Language Understanding**: AI can process and generate human language, enabling tasks like translation, chatbots, and text analysis.

### Types of AI:
- **Narrow AI (Weak AI)**: Designed for specific tasks (e.g., voice assistants like Siri, recommendation systems on Netflix, or image recognition in medical diagnostics).
- **General AI (Strong AI)**: Theoretical, aiming to create systems that can handle any intellectual task a human can perform. This is not yet achieved and remains a goal for future research.

### Applications of AI:
- **Healthcare**: Diagnosing diseases, drug discovery, and personalized treatment plans.
- **Finance**: Fraud detection, algorithmic trading, and risk management.
- **Transportation**: Autonomous vehicles and traffic optimization.
- **Retail**: Customer recommendations, inventory management, and chatbots.
- **Entertainment**: Content personalization, game AI, and virtual assistants.

### Challenges and Ethical Considerations:
- **Bias**: AI systems can inherit biases from training data, leading to unfair outcomes.
- **Privacy**: Data collection and usage raise concerns about user privacy.
- **Job Displacement**: Automation may replace certain jobs, though it also creates new opportunities.
- **Security**: AI can be exploited for malicious purposes, such as deepfakes or cyberattacks.

### How AI Works:
AI systems use algorithms and data to learn patterns and make decisions. Machine learning, a subset of AI, relies on training models on large datasets to improve accuracy over time. For example, a recommendation system learns from user behavior to suggest products or content.

In summary, AI is a transformative technology that enhances efficiency, accuracy, and innovation across industries. While current AI is largely narrow (task-specific), the goal of creating general AI remains a key focus for researchers. As AI evolves, it will continue to shape society, requiring careful consideration of its ethical and societal impacts.

--------------- Performance Metrics ---------------
prompt_tokens: 14
decode_tokens: 1041
time(s): 9.43101
decode_tokens/s_total: 110.381
TTFT (First Token Latency): 121.681 ms
TPOT (Average Token Latency): 8.95111 ms
decode_tokens/s_after_first: 111.718
---------------------------------------------------
```



## Qwen3-4B-AWQ 量化模型推理（BF16 + INT4）

在编译后的 build 目录下，执行以下命令

```bash
cd main
./qwen3_awq_infer /home/yifanfang/LLMInfer/models/qwen3/qwen3_4b_awq_int4_bf16.bin /home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B-AWQ/tokenizer.json
```

在一张 NVIDIA GeForce RTX 4090 上的运行结果

```bash
What is AI?
Qwen3-AWQ-INT4 model warmup...
Qwen3-AWQ-INT4 model generating...
<think>
Okay, the user asked, "What is AI?" I need to explain AI in a clear and concise way. Let me start by defining AI as a general term. Then, I should mention that it's a branch of computer science focused on creating systems that can perform tasks requiring human intelligence. 

I should break down the key components: perception, reasoning, learning, problem-solving, and decision-making. Maybe give examples like machine learning, natural language processing, and robotics. Also, it's important to differentiate AI from general intelligence. I should note that AI is a subset of general intelligence, not the same as human intelligence.

I need to mention the different types of AI, like narrow AI (which is what most systems are today) and general AI (which is still theoretical). Also, touch on the applications of AI in various fields like healthcare, finance, and transportation. Maybe include some challenges or ethical considerations, like bias in algorithms or job displacement.

Wait, the user might be a beginner, so I should avoid jargon. Keep the explanation simple. Also, make sure to structure the answer logically, starting with the definition, then key aspects, types, applications, and challenges. Let me check if I covered all the main points. Yeah, that seems comprehensive. Make sure to end with a summary to reinforce the key points.
</think>

**Artificial Intelligence (AI)** is a branch of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence. These tasks include **perception, reasoning, learning, problem-solving, and decision-making**. AI aims to mimic human cognitive abilities, such as understanding language, recognizing patterns, and adapting to new information.

### Key Aspects of AI:
1. **Types of AI**:
   - **Narrow AI (Weak AI)**: Designed for specific tasks (e.g., voice assistants like Siri, recommendation systems, or image recognition). It excels in one area but lacks general adaptability.
   - **General AI (Strong AI)**: Theoretical AI that could perform any intellectual task a human can. This is still in development and not yet achieved.
   - **Superintelligent AI**: A hypothetical form that surpasses human intelligence, though this remains speculative.

2. **Core Technologies**:
   - **Machine Learning**: Algorithms that improve at tasks through experience (e.g., predicting trends, classifying data).
   - **Natural Language Processing (NLP)**: Enables machines to understand and generate human language (e.g., chatbots, translation tools).
   - **Robotics**: AI-powered systems that perform physical tasks autonomously (e.g., industrial robots, autonomous vehicles).

3. **Applications**:
   - **Healthcare**: Diagnosing diseases, drug discovery, personalized medicine.
   - **Finance**: Fraud detection, algorithmic trading, risk management.
   - **Transportation**: Autonomous vehicles, traffic optimization.
   - **Entertainment**: Content recommendation, game AI, virtual assistants.

4. **Challenges**:
   - **Ethical Concerns**: Bias in algorithms, privacy issues, job displacement.
   - **Technical Limitations**: Current AI struggles with complex, real-world scenarios requiring creativity or emotional intelligence.
   - **Regulation**: Balancing innovation with safety and accountability.

### Why AI Matters:
AI is transforming industries by automating repetitive tasks, improving efficiency, and enabling data-driven decisions. However, its development raises questions about **safety, fairness, and the future of work**. As AI evolves, it will continue to shape society, requiring collaboration between technologists, policymakers, and the public.

In short, **AI is a tool that mimics human intelligence to solve complex problems**, but it is not yet as versatile or self-aware as human cognition.

--------------- Performance Metrics ---------------
prompt_tokens: 14
decode_tokens: 766
time(s): 3.29322
decode_tokens/s_total: 232.599
TTFT (First Token Latency): 56.38 ms
TPOT (Average Token Latency): 4.23091 ms
decode_tokens/s_after_first: 236.356
---------------------------------------------------
```



## 分词器与模型权重下载

### Qwen3-4B 模型（BF16）

从 huggingface 下载模型权重相关文件

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Qwen/Qwen3-4B \
  --local-dir /home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B \
  --local-dir-use-symlinks False
```

将官方模型权重文件导出为 BF16 / FP32 的 C++ 可读 .bin 权重文件

```bash
python tools/export_qwen3/qwen3_dense_unified_converter.py \
  --model_dir /home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B \
  --out_file /home/yifanfang/LLMInfer/models/qwen3/qwen3_4b_fp32.bin
```



### Qwen3-4B-AWQ 模型（BF16 / INT4）

从 huggingface 下载模型权重相关文件

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Qwen/Qwen3-4B-AWQ \
  --local-dir /home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B-AWQ \
  --local-dir-use-symlinks False
```

将官方模型权重文件导出为 BF16 + AWQ INT4 打包的 C++ 可读 .bin 权重文件

```bash
python tools/qwen3_awq/qwen3_awq_converter.py \
  --model_dir /home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B-AWQ \
  --out_file /home/yifanfang/LLMInfer/models/qwen3/qwen3_4b_awq_int4_bf16.bin
```

