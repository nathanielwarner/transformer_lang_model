from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import sentencepiece as spm
from transformer_lm import TransformerLM, predict
import text_data_utils as tdu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("data/csn_java/code_spm.model")
tokenizer.SetEncodeExtraOptions("bos")

print("Loading model...")
model = TransformerLM.from_description("saved_models/delta/model_description.json").to(device)
model.load_state_dict(torch.load("saved_models/delta/trained_model", map_location=device))


PORT = 8001


class CompletionRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.headers['Content-type'] == 'text/plain':
            content_length = int(self.headers['Content-length'])
            body = str(self.rfile.read(content_length))
            print("Received prompt: %s" % body)
            completion = predict(body, model, device, tdu.preprocess_csharp_or_java, tokenizer, 250)
            print("Generated completion: %s" % completion)

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(bytes(completion, 'utf-8'))
        else:
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(bytes('Expected plain text request', 'utf-8'))


def main():
    web_server = HTTPServer(("", PORT), CompletionRequestHandler)
    print("Server started at port %d" % PORT)
    try:
        web_server.serve_forever()
    except KeyboardInterrupt:
        pass
    web_server.server_close()
    print("Server stopped")


if __name__ == "__main__":
    main()
