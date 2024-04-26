import json
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
from tensorrt_llm.builder import get_engine_version


def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)
    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)
    if engine_version is None:
        return config['builder_config']['name'], None
    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if model_arch == 'ChatGLMForCausalLM':
        model_version = config['pretrained_config']['chatglm_version']
    return model_arch, model_version


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   model_name: str = 'llama3',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    use_fast = True
    if tokenizer_type is not None and tokenizer_type == "llama":
        use_fast = False
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                              legacy=False,
                                              padding_side='left',
                                              truncation_side='left',
                                              trust_remote_code=True,
                                              tokenizer_type=tokenizer_type,
                                              use_fast=use_fast)
    if model_name == 'QWenForCausalLM':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config['chat_format']
        if chat_format == 'raw' or chat_format == 'chatml':
            pad_id = gen_config['pad_token_id']
            end_id = gen_config['eos_token_id']
        else:
            raise Exception(f"unknown chat format: {chat_format}")
    elif model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id
