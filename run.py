import torch
import sentencepiece as spm
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from transformer_lm import TransformerLM
import text_data_utils as tdu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("leclair_java/code_spm.model")
tokenizer.SetEncodeExtraOptions("bos")

vocab_size = tokenizer.vocab_size()
d_model = 128
d_ff = 512
n_layers = 4
n_heads = 8
dropout = 0.1
max_sequence_len = 64

model = TransformerLM(vocab_size, max_sequence_len, d_model, n_heads, d_ff, n_layers, dropout=dropout).to(device)

model_save_path = "trained_model"

print("Loading existing model...")
model.load_state_dict(torch.load(model_save_path))


while True:
    model.eval()
    prompt = input("Prompt: ")
    if prompt == "exit" or prompt == "quit":
        break
    prompt = tdu.preprocess_csharp_or_java(prompt)
    prompt_tok = tokenizer.EncodeAsIds(prompt)
    len_prompt = len(prompt_tok)
    prompt_padded = pad_sequences([prompt_tok], maxlen=max_sequence_len, padding='post', truncating='post', value=0,
                                  dtype='int64')
    out_toks = []
    for j in range(50):
        out = model(torch.tensor(prompt_padded).to(device))[0]
        working_idx = min(len_prompt - 1 + j, max_sequence_len - 1)
        new_tok = torch.argmax(out[working_idx]).item()
        out_toks.append(new_tok)
        if new_tok == tokenizer.eos_id():
            break
        if len_prompt + j < max_sequence_len:
            prompt_padded[0][len_prompt + j] = new_tok
        else:
            prompt_padded = np.roll(prompt_padded, -1, axis=1)
            prompt_padded[0][-1] = new_tok
    out_detok = tokenizer.DecodeIds(out_toks)
    print("Model completion: %s" % out_detok)
