import gc
import json
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import argparse
import torch
from tensorrt_llm_study.runtime import ModelRunner, PYTHON_BINDINGS
from tensorrt_llm_study.logger import logger
from utils import load_tokenizer, read_model_name, throttle_generator

if PYTHON_BINDINGS:
    from tensorrt_llm_study.runtime import ModelRunnerCpp

runner = None
tokenizer = None

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000


def parse_arguments():
    parser = argparse.ArgumentParser(description="Interactive TensorRT Model Chat Interface")
    parser.add_argument('--engine_dir', type=str, default="",
                        help="Directory of the TensorRT engine")
    parser.add_argument('--tokenizer_dir', type=str, default="",
                        help="Directory of the tokenizer")
    parser.add_argument('--log_level', type=str, default='info', help="Logging level")
    parser.add_argument('--use_py_session', default=True, help="Flag to use Python session for inference")
    parser.add_argument('--streaming', default=True, help="Flag to use Python session for inference")
    parser.add_argument('--streaming_interval', type=int, default=5, help="Interval for streaming inference")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature setting for generation")
    parser.add_argument('--num_beams', type=int, default=1, help="Number of beams for beam search")
    parser.add_argument('--medusa_choices', type=str, default=None,
                        help="Medusa choice to use, if not none, will use Medusa decoding."
                             "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
                        )
    parser.add_argument('--end_id', type=int, default=128009, help="End token ID")  # for llama set 128009

    return parser.parse_args()


def setup_model(args):
    logger.set_level(args.log_level)
    global runner, tokenizer
    model_name, model_version = read_model_name(args.engine_dir)
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        model_name=model_name,
        model_version=model_version
    )

    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner = runner_cls.from_dir(engine_dir=args.engine_dir)


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
    input_lengths = input_ids.size(1)
    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=input_ids,
            end_id=tokenizer.eos_token_id if args.end_id is None else args.end_id,
            pad_id=tokenizer.pad_token_id,
            temperature=temperature,
            max_beam_width=args.num_beams,
            streaming=args.streaming,
            top_k=1,
            top_p=top_p,
            repetition_penalty=1.0,
            presence_penalty=1.0,
            frequency_penalty=1.0,
            max_new_tokens=max_new_tokens,
            output_sequence_lengths=True,
            return_dict=True,
            medusa_choices=args.medusa_choices)

        torch.cuda.synchronize()

    for curr_outputs in throttle_generator(outputs, args.streaming_interval):
        output_ids = curr_outputs['output_ids'][0][0]
        sequence_length = curr_outputs['sequence_lengths'][0][0]
        start = max(input_lengths, sequence_length - args.streaming_interval)
        new_output_ids = output_ids[start:sequence_length]
        response = tokenizer.decode(new_output_ids, skip_special_tokens=True)
        yield {
            "text": response,
            "usage": {
                "prompt_tokens": input_lengths,
                "completion_tokens": sequence_length - input_lengths,
                "total_tokens": sequence_length
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
    logger.debug(f"==== request ====\n{gen_params}")
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
        logger.debug(f"==== message ====\n{message}")
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
    setup_model(args)

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
