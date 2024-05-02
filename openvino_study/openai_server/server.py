import gc
import json
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Literal
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import argparse
from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino.utils import OV_XML_FILE_NAME
from transformers import (PretrainedConfig, AutoTokenizer, TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)

from typing import Optional, Union, Dict
from pathlib import Path
from threading import Thread
import torch

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Interactive TensorRT Model Chat Interface")
    parser.add_argument('--model_path', type=str, default="", help="Directory of the OpenVINO model")
    parser.add_argument('--tokenizer_dir', type=str, default="", help="Directory of the tokenizer")
    parser.add_argument('--log_level', type=str, default='info', help="Logging level")
    parser.add_argument('--use_py_session', default=True, help="Flag to use Python session for inference")
    parser.add_argument('--device', default='CPU', type=str, help='Device for inference')
    parser.add_argument('--end_id', type=int, default=128009, help="End token ID")  # for llama set 128009
    return parser.parse_args()


class OVCHATModel(OVModelForCausalLM):

    def __init__(
            self,
            model: "Model",
            config: "PretrainedConfig" = None,
            device: str = "CPU",
            dynamic_shapes: bool = True,
            ov_config: Optional[Dict[str, str]] = None,
            model_save_dir: Optional[Union[str, Path]] = None,
            **kwargs,
    ):
        NormalizedConfigManager._conf["chatglm"] = NormalizedTextConfig.with_args(
            num_layers="num_hidden_layers",
            num_attention_heads="num_attention_heads",
            hidden_size="hidden_size",
        )
        super().__init__(
            model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs
        )

    def _reshape(
            self,
            model: "Model",
            *args, **kwargs
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            input_name = inputs.get_any_name()
            if input_name.startswith('beam_idx'):
                continue
            if input_name.startswith('past_key_values'):
                shapes[inputs][1] = -1
                shapes[inputs][2] = 2
            elif shapes[inputs].rank.get_length() > 1:
                shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    @classmethod
    def _from_pretrained(
            cls,
            model_id: Union[str, Path],
            config: PretrainedConfig,
            use_auth_token: Optional[Union[bool, str, None]] = None,
            revision: Optional[Union[str, None]] = None,
            force_download: bool = False,
            cache_dir: Optional[str] = None,
            file_name: Optional[str] = None,
            subfolder: str = "",
            from_onnx: bool = False,
            local_files_only: bool = False,
            load_in_8bit: bool = False,
            **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path)
        init_cls = OVCHATModel

        return init_cls(
            model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # clean cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: CompletionUsage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(
        id="MiniCPM-2B"
    )
    return ModelList(
        data=[model_card]
    )


def generate_model(params: dict):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))

    input_ids = tokenizer.apply_chat_template(messages,
                                              add_generation_prompt=True,
                                              tokenize=True,
                                              return_tensors="pt")
    input_lengths = input_ids.shape[1]
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": input_ids,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens([args.end_id])]),
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    for response in streamer:
        if response:
            output_len = len(tokenizer.encode(response))
            yield {
                "text": response,
                "usage": {
                    "prompt_tokens": input_lengths,
                    "completion_tokens": output_len,
                    "total_tokens": input_lengths + output_len
                },
                "finish_reason": "",
            }
    gc.collect()
    torch.cuda.empty_cache()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 2048,
        echo=False,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
    )
    print(f"==== request ====\n{gen_params}")
    input_tokens = sum(len(tokenizer.encode(msg.content)) for msg in request.messages)
    if request.stream:
        async def stream_response():
            previous_text = ""
            for new_response in generate_model(gen_params):
                delta_text = new_response["text"][len(previous_text):]
                previous_text = new_response["text"]
                delta = DeltaMessage(content=delta_text, role="assistant")
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None
                )
                chunk = {
                    "model": request.model,
                    "id": "",
                    "choices": [choice_data.dict(exclude_none=True)],
                    "object": "chat.completion.chunk"
                }
                yield json.dumps(chunk) + "\n"

        return EventSourceResponse(stream_response(), media_type="text/event-stream")

    else:
        generated_text = ""
        for response in generate_model(gen_params):
            generated_text += response["text"]
        generated_text = generated_text.strip()
        output_tokens = len(tokenizer.encode(generated_text))
        usage = UsageInfo(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=output_tokens + input_tokens
        )
        message = ChatMessage(role="assistant", content=generated_text)
        print(f"==== message ====\n{message}")
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop",
        )
        return ChatCompletionResponse(
            model=request.model,
            id="",
            choices=[choice_data],
            object="chat.completion",
            usage=usage
        )


if __name__ == "__main__":
    global args
    args = parse_arguments()
    model = OVCHATModel.from_pretrained(
        model_id=args.model_path,
        device=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
