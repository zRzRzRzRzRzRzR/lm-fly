# vLLM 加速套件

## 工具介绍

* [vLLM](https://github.com/vllm-project/vllm) 是加州伯克利大学LMSYS组织开源的大语言模型高速推理框架，旨在极大地提升实时场景下的语言模型服务的吞吐与内存使用效率。
* vLLM是一个快速且易于使用的库，用于LLM推理和服务。vLLM利用了全新的注意力算法PagedAttention，有效地管理注意力键和值。在吞吐量方面，vLLM的性能比HuggingFace Transformers (HF)高出24倍。
为了最大化性能并减少内存占用，vLLM支持Continuous batching of incoming requests高并发批推理机制，其实现是在1个独立线程中运行推理并且对用户提供请求排队合批机制，能够满足在线服务的高吞吐并发服务能力。vLLM提供asyncio封装，在主线程中基于uvicorn+fastapi封装后的asyncio http框架，可以实现对外HTTP接口服务，并将请求提交到vLLM的队列进入到vLLM的推理线程进行continuous batching批量推理，主线程异步等待推理结果，并将结果返回到HTTP客户端。
* vLLM支持多种量化技术，包括GPTQ, AWQ, SqueezeLLM, FP8 KV Cache，以进一步提升推理速度和内存效率。

## 安装vLLM

### 安装官方仓库

本仓库使用的是`vLLM-0.4.1`版本进行实验。确保您安装了正确的发行版。
```
pip install vllm==0.4.1
```

### 安装额外依赖

您需要安装额外的依赖来支持本章节的所有功能。
```shell
pip install -r ../requirements.txt
```
## 使用 OpenAI 接口

### 快速使用

**vLLM实现了一个基于FastAPI的对OpenAI接口兼容的HTTP服务，可以通过Python运行模块的方式直接启动该服务**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct # 权重路径或HuggingFace模型名称
  --served-model-name llama3 # OpenAI接口中使用的model名称
```

多卡运行（此项依赖Ray框架，可能需要额外安装）
```bash
export NCCL_P2P_DISABLE=1 # 如果在不支持NVlink的机器上运行以下指令异常，建议添加此环境变量
python -m vllm.entrypoints.openai.api_server \
--model meta-llama/Meta-Llama-3-8B-Instruct # 权重路径或HuggingFace模型名称
--served-model-name llama3 # OpenAI接口中使用的model名称
--max-num-seqs 5 # 最大并发数
--max-model-len 8192 # 模型推理的最大长度
--tensor-parallel-size 2 # 将模型加载在多个GPU上
--gpu-memory-utilization 0.95 # GPU内存利用率(该值影响KV Cache的序列长度，进而影响最大并发数和模型最大长度，默认值为0.9)
```

您可以运行 [openai_client.py](openai_server/request.py) 来测试服务是否正常。

```bash
python openai_server/request.py
```

### 进阶使用

**参考官方文档的接口参数[vLLM engine args](https://docs.vllm.ai/en/latest/models/engine_args.html)**
```
TO DO: 针对官方文档的参数进行详细解释
```

### 局限性

1. 不支持CPU推理
2. LoRA模块推理不如HuggingFace Transformers稳定
3. 可能存在与HuggingFace Transformers推理结果不一致的情况

## 使用 WebUI 接口

```python
python webui_infer.py -m llama3 \
--model-url http://127.0.0.1:8000/v1 \
--host 0.0.0.0 \
--port 8002 \
--stop-token-ids 128001
```

## 额外说明

vLLM支持离线推理的方式，同样地，你也可以使用vLLM的离线推理方式来进行推理进而实现你自己定制的API Server。
通常情况下，Model as a Service(MaaS)的方式更适合于在线推理，而离线推理则更适合于批量推理的场景，在此不做赘述。