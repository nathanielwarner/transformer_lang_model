import os
import sys
import torch
import sentencepiece as spm
from transformer_lm import TransformerLM, predict
import text_data_utils as tdu


if len(sys.argv) != 3:
    print("Expected 2 arguments (path to model, path to dataset)")
    exit(1)

model_path = os.path.abspath(sys.argv[1])
dataset_path = os.path.abspath(sys.argv[2])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(os.path.join(dataset_path, "code_spm.model"))
tokenizer.SetEncodeExtraOptions("bos")

print("Loading model...")
model = TransformerLM.from_description(os.path.join(model_path, "model_description.json")).to(device)
model.load_state_dict(torch.load(os.path.join(model_path, "trained_model"), map_location=device))


while True:
    model.eval()
    prompt = input("Prompt: ")
    if prompt == "exit" or prompt == "quit":
        break
    result = predict(prompt, model, device, tdu.preprocess_csharp_or_java, tokenizer, 250)
    print("Model completion: %s" % result)
