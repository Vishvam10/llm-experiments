import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def get_hidden_states(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # (batch_size, sequence_length) with real tokens = 1 and padding tokens = 0
    token_mask = inputs["attention_mask"]

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # Tuple of len = num_layers  :(batch_size, sequence_length, hidden_dim)
    hidden_states = out.hidden_states

    # Gives number of valid (non-padding) tokens per sequence : (batch_size, )
    token_lengths = token_mask.sum(dim=1)

    # Index of last non-padding token for each sequence : (batch_size, )
    last_token_idx = token_lengths - 1

    reps = []

    for layer in hidden_states:
        # Since batch_size = 1, directly index instead of keeping batch dimension

        # For each batch item, pick the hidden state at its last valid token
        # (hidden_dim)
        selected = layer[0, last_token_idx.item()]

        # (hidden_dim)
        cpu_tensor = selected.cpu()
        numpy_array = cpu_tensor.numpy()

        reps.append(numpy_array)

    # Stacking along new layers : (num_layers, hidden_dim)
    stacked = np.stack(reps)

    return stacked