import json
import random
import re
import tqdm


def preprocess(x: str, remove_stars=False, remove_java_doc_vars=False, remove_html_tags=False, remove_comments=False,
               remove_start_and_end_quotes=False, lower=False) -> str:
    if remove_java_doc_vars:
        x = re.sub(r'(?<![{])(@[\s\S]*)', ' ', x)
    if remove_comments:
        x = re.sub(r'(?<![:\"])(//.*?(?:\n|\\n))', ' ', x)
    if remove_html_tags:
        x = re.sub(r'<.*?>', ' ', x)
    x = x.replace('\\n', ' ').replace('\n', ' ')
    x = x.replace('\\t', ' ').replace('\t', ' ')
    if remove_stars:
        x = x.replace('/*', ' ').replace('*/', ' ').replace('*', ' ')
    if remove_start_and_end_quotes:
        x = x.strip()
        if x.startswith("'"):
            x = x[len("'"):]
        if x.endswith("'"):
            x = x[:-len("'")]
        if x.startswith('"'):
            x = x[len('"'):]
        if x.endswith('"'):
            x = x[:-len('"')]
    x = x.strip()
    x = re.sub(r'(\s\s+)', ' ', x)
    if lower:
        x = x.lower()
    return x


def preprocess_java(x: str) -> str:
    return preprocess(x, remove_comments=True, remove_start_and_end_quotes=True)


def preprocess_javadoc(x: str) -> str:
    return preprocess(x, remove_stars=True, remove_java_doc_vars=True, remove_html_tags=True)


def write_to_files(items, filename_prefix):
    print("Writing %s..." % filename_prefix)
    out_codes_f = open(filename_prefix + "_codes.txt", mode='w', encoding='utf-8')
    out_nl_f = open(filename_prefix + "_nl.txt", mode='w', encoding='utf-8')
    for ex in tqdm.tqdm(items):
        ex_code = preprocess_java(ex[1])
        ex_nl = preprocess_javadoc(ex[0])
        out_codes_f.write(ex_code + "\n")
        out_nl_f.write(ex_nl + "\n")


def main():

    print("Loading files...")
    with open("fid_pid.json", mode='r', encoding='utf-8') as fid_pid_file:
        fid_pid = json.load(fid_pid_file)

    with open("comments.json", mode='r', encoding='utf-8') as summaries_file:
        summaries = json.load(summaries_file)

    with open("functions.json", mode='r', encoding='utf-8') as codes_file:
        codes = json.load(codes_file)

    print("Splitting into train/val/test...")
    assert len(summaries) == len(codes)
    num_examples = len(summaries)
    num_train = 14 * num_examples / 16
    num_val = num_examples / 16
    num_test = num_examples / 16

    fids = list(summaries.keys())
    random.seed(a=420)
    random.shuffle(fids)

    train = []
    val = []
    test = []

    train_pids = []
    val_pids = []
    test_pids = []

    for fid in tqdm.tqdm(fids):
        pid = fid_pid[fid]
        item = (summaries[fid], codes[fid], pid)
        if pid in train_pids:
            train.append(item)
        elif pid in val_pids:
            val.append(item)
        elif pid in test_pids:
            test.append(item)
        else:
            possible_sets = []
            if len(train) < num_train:
                for _ in range(14):
                    possible_sets.append("train")
            if len(val) < num_val:
                possible_sets.append("val")
            if len(test) < num_test:
                possible_sets.append("test")
            which_set = random.choice(possible_sets)
            if which_set == "train":
                train.append(item)
                train_pids.append(pid)
            elif which_set == "val":
                val.append(item)
                val_pids.append(pid)
            elif which_set == "test":
                test.append(item)
                test_pids.append(pid)
            else:
                raise Exception()

    write_to_files(train, "train")
    write_to_files(val, "val")
    write_to_files(test, "test")


if __name__ == "__main__":
    main()
