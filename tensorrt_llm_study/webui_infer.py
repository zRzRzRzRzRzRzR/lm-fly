"""gradio demo"""

import argparse
import gradio as gr
import torch
from tensorrt_llm_study.runtime import ModelRunner, PYTHON_BINDINGS
from tensorrt_llm_study.logger import logger
from utils import load_tokenizer, read_model_name, throttle_generator

if PYTHON_BINDINGS:
    from tensorrt_llm_study.runtime import ModelRunnerCpp

runner = None
tokenizer = None


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


def chat_function(history, max_length, top_p, temperature):
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

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
            max_new_tokens=max_length,
            output_sequence_lengths=True,
            return_dict=True,
            medusa_choices=args.medusa_choices)

        torch.cuda.synchronize()
    if args.streaming:
        for curr_outputs in throttle_generator(outputs, args.streaming_interval):
            output_ids = curr_outputs['output_ids'][0][0]
            sequence_length = curr_outputs['sequence_lengths'][0][0]
            start = max(input_lengths, sequence_length - args.streaming_interval)
            new_output_ids = output_ids[start:sequence_length]
            response = tokenizer.decode(new_output_ids, skip_special_tokens=True)
            history[-1][1] += response

            yield history


def main():
    global args
    args = parse_arguments()
    setup_model(args)

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center"> TensorRT-LLM gradio by lm-fly </h1>""")
        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
                submitBtn = gr.Button("Submit")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 8192, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)

        def user(query, history):
            return "", history + [[query, ""]]

        submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            chat_function, [chatbot, max_length, top_p, temperature], chatbot
        )
        emptyBtn.click(lambda: None, None, chatbot, queue=False)
    demo.launch(server_name="0.0.0.0", server_port=7863, inbrowser=False, share=True)


if __name__ == '__main__':
    main()
