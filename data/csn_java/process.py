import os
import gzip
import jsonlines
import re
import sentencepiece as spm


def preprocess(code):
    code = re.sub(r' {4}', r'\\t', code)
    code = re.sub(r'\\n\\t', r'\\n', code)
    return code


def extract_codes(dir_path, out_filename):
    if not os.path.isfile(out_filename):
        codes = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".jsonl.gz"):
                full_filename = os.path.join(dir_path, filename)
                print(full_filename)
                with gzip.open(full_filename, mode="rb") as file:
                    for example in jsonlines.Reader(file):
                        code = preprocess(example["code"].encode("unicode_escape").decode("utf-8"))
                        codes.append(code)
        out_write = "\n".join(codes)
        with open(out_filename, mode="w", encoding="utf-8") as out_file:
            out_file.write(out_write)
    else:
        print("Skipping target %s because it already exists" % out_filename)


extract_codes("train", "train_codes.txt")
extract_codes("valid", "val_codes.txt")
extract_codes("test", "test_codes.txt")

spm.SentencePieceTrainer.Train(input="train_codes.txt", model_prefix="code_spm", model_type="unigram", vocab_size=8192,
                               character_coverage=1.0, user_defined_symbols=["\\n", "\\t"])
