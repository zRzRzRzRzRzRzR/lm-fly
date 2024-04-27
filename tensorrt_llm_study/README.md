# TensorRT-LLM 加速套件

## 工具介绍

* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 是英伟达提供的一个大模型加速框架。
* TensorRT-LLM为用户提供了一个易于使用的Python API，用于定义大型语言模型（LLMs）并构建包含最先进优化的TensorRT引擎，以在NVIDIA
  GPU上高效执行推理。TensorRT-LLM还包含创建执行这些TensorRT引擎的Python和C++运行时的组件。
* 为了最大化性能并减少内存占用，TensorRT-LLM允许使用不同的量化模式执行模型（参考 support matrix ）。TensorRT-LLM支持 INT4 或
  INT8 权重（以及 FP16 激活；也就是仅限 INT4/INT8 权重）以及 SmoothQuant 技术的完整实现。

## 制作引擎

### 安装官方镜像

本仓库使用的是 `TensorRT-LLM-0.9.0` 版本进行实验。确保您安装正确的正确的发行版。
```shell
wget https://github.com/NVIDIA/TensorRT-LLM/archive/refs/tags/v0.9.0.zip
```

构建大模型的TensorRT-LLM 引擎并不是一个容易的事情，以 Llama3 为例，在[官方教程](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows/examples/llama)中，您需要自行构建docker，并使用教程的方法转换模型。

**这一步是没有办法跳过的**。

### 安装额外依赖

您需要安装额外的依赖来支持本章节的所有功能。
```sheel
pip install -r ../requirements.txt
```
## 使用 OpenAI 接口

### 快速使用

**请注意，如果您有多张显卡，一定需要指定显卡运行。**

```python
CUDA_VISIBLE_DEVICES = 0 python openai_server/server.py
```

您可以运行 [openai_client.py](openai_server/request.py) 来测试服务是否正常。

```python
python openai_server/server.py
```

### 局限性

1. 目前，示例代码并不支持使用工具调用和 embedding 模型。

## 使用 WebUI 接口

**请注意，如果您有多张显卡，一定需要指定显卡运行。**

```python
CUDA_VISIBLE_DEVICES = 0 python webui_infer.py
```
之后，您可以在gradio页面上进行访问，效果如下:

![TRT](assets/trt_gradio.gif)

