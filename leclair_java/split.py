import json
import sys
sys.path.append("..")
import text_data_utils as tdu


def do_split(in_filename, out_code_filename, out_nl_filename):
    in_file = open(in_filename, "r", encoding="utf-8")
    code = []
    nl = []
    for line in in_file.readlines():
        ex = json.loads(line)
        ex_code = tdu.preprocess_csharp_or_java(repr(ex["code"]))
        ex_nl = tdu.preprocess_javadoc(repr(ex["nl"]))
        if len(ex_code) > 4 and len(ex_nl) > 4:
            code.append(ex_code + "\n")
            nl.append(ex_nl + "\n")
    out_code_file = open(out_code_filename, "w", encoding="utf-8")
    out_nl_file = open(out_nl_filename, "w", encoding="utf-8")
    out_code_file.writelines(code)
    out_nl_file.writelines(nl)
    in_file.close()
    out_nl_file.close()
    out_code_file.close()


do_split("train.json", "train_codes.txt", "train_nl.txt")
do_split("val.json", "val_codes.txt", "val_nl.txt")
do_split("test.json", "test_codes.txt", "test_nl.txt")
