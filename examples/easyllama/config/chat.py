# chat with a model
# good for debugging and playing on macbooks and such

# tokenizer
model_name_or_path = 'NousResearch/Llama-2-7b-hf'

# model
model_name = 'EasyLlama_120M'
pretrained_path = 'out/alpaca-chat-120m/checkpoint-2/'

# chat
block_size = 1024
do_sample=True
top_p=0.95
top_k=50
temperature=0.7
num_beams=1
