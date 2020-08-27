import os
import torch
import sentencepiece as spm
from transformer_lm import TransformerLM, predict
import text_data_utils as tdu


dataset_path = os.path.abspath(os.getenv("DATASET_PATH"))
model_path = os.path.abspath(os.getenv("MODEL_PATH"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(os.path.join(dataset_path, "code_spm.model"))
tokenizer.SetEncodeExtraOptions("bos")

print("Loading model...")
model = TransformerLM.from_description(os.path.join(model_path, "model_description.json")).to(device)
model.load_state_dict(torch.load(os.path.join(model_path, "trained_model"), map_location=device))


def get_completion(environ, start_response):

    if "CONTENT_TYPE" not in environ or environ["CONTENT_TYPE"] != "text/plain":
        error = b"Expected plain text"
        start_response("400 Bad Request", [
            ("Content-Type", "text/plain"),
            ("Content_Length", str(len(error)))
        ])
        return iter([error])

    if "CONTENT_LENGTH" not in environ:
        error = b"Content length was not defined"
        start_response("400 Bad Request", [
            ("Content-Type", "text/plain"),
            ("Content-Length", str(len(error)))
        ])
        return iter([error])

    request_body_size = int(environ["CONTENT_LENGTH"])
    request_body = str(environ["wsgi.input"].read(request_body_size))
    print("Prompt: %s" % request_body)

    completion = bytes(predict(request_body, model, device, tdu.preprocess_csharp_or_java, tokenizer, 20), "utf-8")
    print("Completion: %s\n" % completion)

    start_response("200 OK", [
        ("Content-Type", "text/plain"),
        ("Content_Length", str(len(completion)))
    ])
    return iter([completion])
