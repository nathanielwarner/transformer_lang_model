import json
import random


with open("fid_pid.json", mode='r', encoding='utf-8') as fid_pid_file:
    fid_pid = json.load(fid_pid_file)

with open("comments.json", mode='r', encoding='utf-8') as summaries_file:
    summaries = json.load(summaries_file)

with open("functions.json", mode='r', encoding='utf-8') as codes_file:
    codes = json.load(codes_file)

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

for fid in fids:
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


def write_to_file(items, filename):
    out_file = open(filename, mode='w', encoding='utf-8')
    for item in items:
        out_json = json.dumps({"code": item[1], "nl": item[0]})
        out_file.write(out_json + "\n")


write_to_file(train, "train.json")
write_to_file(val, "val.json")
write_to_file(test, "test.json")
