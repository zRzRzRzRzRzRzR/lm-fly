import argparse
import gradio as gr
from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino.utils import OV_XML_FILE_NAME
from transformers import (PretrainedConfig, AutoTokenizer, TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)

from typing import Optional, Union, Dict
from pathlib import Path
from threading import Thread
import torch

global model, tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Interactive TensorRT Model Chat Interface")
    parser.add_argument('--model_path', type=str, default="", help="Directory of the OpenVINO model")
    parser.add_argument('--tokenizer_path', type=str, default="",help="Directory of the tokenizer")
    parser.add_argument('--log_level', type=str, default='info', help="Logging level")
    parser.add_argument('--use_py_session', default=True, help="Flag to use Python session for inference")
    parser.add_argument('--device', default='CPU', type=str, help='Device for inference')
    parser.add_argument('--end_id', type=int, default=128009, help="End token ID")  # for llama set 128009
    return parser.parse_args()


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


def predict(history, max_length, top_p, temperature):
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    print("\n\n====conversation====\n", messages)
    model_inputs = tokenizer.apply_chat_template(messages,
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens([args.end_id])]),
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    for new_token in streamer:
        if new_token != '':
            history[-1][1] += new_token
            yield history


if __name__ == "__main__":
    args = parse_arguments()
    model = OVCHATModel.from_pretrained(
        model_id=args.model_path,
        device=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">OpenVINO infer Gradio</h1>""")
        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)


        def user(query, history):
            return "", history + [[query, ""]]


        submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            predict, [chatbot, max_length, top_p, temperature], chatbot
        )
        emptyBtn.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8000, inbrowser=False, share=True)
