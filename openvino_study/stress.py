"""
压力测试菜吗，测试以下内容：
1. 模型的首个token生成时间
2. 模型的总生成时间
3. 模型的平均生成速度
"""
import argparse
import time
from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino.utils import OV_XML_FILE_NAME

from transformers import (PretrainedConfig, AutoTokenizer, AutoConfig,
                          TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)

from typing import Optional, Union, Dict
from pathlib import Path
from threading import Thread
import torch

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


class OVCHATGLMModel(OVModelForCausalLM):
    """
    Optimum intel compatible model wrapper for CHATGLM2
    """

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
            local_files_only: bool = False,
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
        init_cls = OVCHATGLMModel

        return init_cls(
            model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs
        )


def generate_random_input(tokenizer, input_length):
    vocab_size = len(tokenizer.get_vocab())
    tokens = torch.randint(0, vocab_size, (1, input_length))
    return tokens


def perform_stress_test(model, tokenizer, input_length=1024, output_length=1024, iterations=1):
    times_first_token = []
    times_total = []
    tokens_generated = []

    # 生成input_length长度的token

    for _ in range(iterations):
        model_inputs = generate_random_input(tokenizer, input_length)
        token_out = True
        total_tokens = 0
        streamer = TextIteratorStreamer(
            tokenizer, timeout=300.0, skip_prompt=True, skip_special_tokens=True
        )
        stop_tokens = [0, 1, 2]
        stop_tokens = [StopOnTokens(stop_tokens)]
        generate_kwargs = dict(
            input_ids=model_inputs,
            max_new_tokens=output_length,
            temperature=0.1,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList(stop_tokens)
        )
        start_time = time.time()
        t1 = Thread(target=ov_model.generate, kwargs=generate_kwargs)
        t1.start()
        for new_text in streamer:
            if token_out:
                first_token_time = time.time() - start_time
                times_first_token.append(first_token_time)
                token_out = False
            total_tokens += len(tokenizer.encode(new_text))

        end_time = time.time()
        total_time = end_time - start_time

        times_total.append(total_time)
        tokens_generated.append(total_tokens)

    avg_first_token = sum(times_first_token) / len(times_first_token)
    avg_total_time = sum(times_total) / len(times_total)
    avg_tokens_per_second = sum(tokens_generated) / sum(times_total)

    print(f"Average time for first token: {avg_first_token:.2f} seconds")
    print(f"Average total time per input: {avg_total_time:.2f} seconds")
    print(f"Average tokens per second: {avg_tokens_per_second:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test for model.")
    parser.add_argument('--model_path', type=str, default="./model", help="Path to the model directory.")
    parser.add_argument('--input_length', default=1024, type=int, help="Maximum length of input sequence.")
    parser.add_argument('--output_length', default=1024, type=int, help="Maximum length of output sequence.")
    parser.add_argument('--iterations', default=1, type=int, help="Number of test iterations to perform.")
    parser.add_argument('--device', default='CPU', required=False, type=str, help='Required. device for inference')
    args = parser.parse_args()
    ov_config = {"PERFORMANCE_HINT": "LATENCY",
                 "NUM_STREAMS": "1", "CACHE_DIR": ""}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    model_dir = args.model_path

    print("====Compiling model====")
    ov_model = OVCHATGLMModel.from_pretrained(
        model_dir,
        device=args.device,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )
    print("====Testing====")
    perform_stress_test(
        model=ov_model,
        tokenizer=tokenizer,
        input_length=args.input_length,
        output_length=args.output_length,
        iterations=args.iterations
    )
