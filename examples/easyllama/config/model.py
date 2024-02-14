ModelConfigs = {}


# Easy-Llama
ModelConfigs["EasyLlama_10M"] = {    # 10M for debugging
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "max_position_embeddings": 2048,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "attention_dropout": 0.0,
    "_attn_implementation": "sdpa",
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 32000
}
ModelConfigs["EasyLlama_120M"] = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "max_position_embeddings": 2048,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_key_value_heads": 12,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "attention_dropout": 0.0,
    "_attn_implementation": "sdpa",
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 32000
}
ModelConfigs["EasyLlama_1B"] = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 5632,
    "max_position_embeddings": 4096,
    "num_hidden_layers": 22,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "attention_dropout": 0.0,
    "_attn_implementation": "sdpa",
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 32000
}

