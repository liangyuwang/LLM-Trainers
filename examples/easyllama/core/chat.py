# coppied from Tiny-Llama

import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    TextStreamer,
)
from threading import Thread


exec(open('utils/configurator.py').read()) # overrides from command line or config file
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list, type(None)))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
def load_args(args, config: dict):
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

# Loading the tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
model = LlamaForCausalLM.from_pretrained(config['pretrained_path'])

# using CUDA for an optimal experience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Defining a custom stopping criteria class for the model's text generation.
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                return True
        return False


# Function to generate model predictions.
def predict_online(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    # Formatting the input for the model.
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=config['block_size'],
        do_sample=config['do_sample'],
        top_p=config['top_p'],
        top_k=config['top_k'],
        temperature=config['temperature'],
        num_beams=config['num_beams'],
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
            break
        yield partial_message


# Function to generate model predictions for CLI.
def predict_cli(message, history):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    past_key_values = None

    seq_len = 0
    while True:
        print("User: ", end="")
        user_input = input()
        print("\n")

        user_entry = dict(role="user", content=user_input)
        input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

        if past_key_values is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
            attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

        print("ChatBot: ", end="")
        result = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            streamer=streamer,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        print("\n")

        past_key_values = result["past_key_values"]


if __name__ == "__main__":
    use_cli = input("Do you want to use the website as interface? (yes/no): ")
    if use_cli.lower() == 'no':
        print("You are now in the CLI mode. Type 'exit' to quit.")
        predict_cli()
    else:
        # Setting up the Gradio chat interface.
        gr.ChatInterface(predict_online,
                        title="Privacyllama_chatBot",
                        description="Ask Privacy llama any questions",
                        examples=['How to cook a fish?', 'Who is the president of US now?']
                        ).launch()  # Launching the web interface.

