import torch
import sentencepiece as spm
from transformer_lm import TransformerLM, predict
import text_data_utils as tdu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("leclair_java/code_spm.model")
tokenizer.SetEncodeExtraOptions("bos")

print("Loading model...")
model = TransformerLM.from_description("saved_models/beta/model_description.json").to(device)
model.load_state_dict(torch.load("saved_models/beta/trained_model", map_location=device))


while True:
    model.eval()
    prompt = input("Prompt: ")
    if prompt == "exit" or prompt == "quit":
        break
    result = predict(prompt, model, device, tdu.preprocess_csharp_or_java, tokenizer, 250)
    print("Model completion: %s" % result)
