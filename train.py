import time
import os
import sys
import torch
import tqdm
import sentencepiece as spm
import random
from keras_preprocessing.sequence import pad_sequences

from transformer_lm import TransformerLM

if len(sys.argv) != 3:
    print("Expected 2 arguments (path to model, path to dataset)")
    exit(1)

model_path = os.path.abspath(sys.argv[1])
dataset_path = os.path.abspath(sys.argv[2])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(filename):
    file = open(filename, "r", encoding="utf-8")
    examples = []
    for line in file:
        line = line[:-len("\n")]
        examples.append(line)
    return examples


tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(os.path.join(dataset_path, "code_spm.model"))
tokenizer.SetEncodeExtraOptions("bos:eos")


print("Loading dataset...")

leclair_train = load(os.path.join(dataset_path, "train_codes.txt"))
leclair_val = load(os.path.join(dataset_path, "val_codes.txt"))

print("Creating model...")

model = TransformerLM.from_description(os.path.join(model_path, "model_description.json")).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

batch_size = 80

random.seed()


def pad_and_split(batch, max_item_len):
    padded_batch = pad_sequences(batch, maxlen=max_item_len, padding='post', truncating='post',
                                 value=0, dtype='int64')
    padded_batch = torch.tensor(padded_batch).to(device)
    context = padded_batch[:, :-1]
    target = padded_batch[:, 1:]
    return context, target


def batch_dataset(dataset):
    random.shuffle(dataset)
    num_batches = (len(dataset) - 1) // batch_size
    max_item_len = model.input_length + 1
    for batch_num in tqdm.trange(num_batches):
        batch = dataset[batch_num * batch_size: batch_num * batch_size + batch_size]
        tokenized_batch = []
        for item in batch:
            tokenized = tokenizer.SampleEncodeAsIds(item, -1, 0.2)
            if len(tokenized) <= max_item_len:
                tokenized_batch.append(tokenized)
        yield pad_and_split(tokenized_batch, max_item_len)


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()

    counter = 0

    for context, target in batch_dataset(leclair_train):

        counter += 1

        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output.view(-1, model.vocab_size), target.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if counter % log_interval == 0 and counter > 0:
            cur_loss = total_loss / counter
            elapsed = time.time() - start_time
            print("Batch %d, current train loss: %.4f, time elapsed: %.1f"
                  % (counter, cur_loss, elapsed))


def evaluate(data_source):
    model.eval()
    total_loss = 0.
    batch_counter = 0
    with torch.no_grad():
        for context, target in batch_dataset(data_source):
            output = model(context)
            total_loss += criterion(output.view(-1, model.vocab_size), target.flatten()).item()
            batch_counter += 1
    return total_loss / batch_counter


model_save_path = os.path.join(model_path, "trained_model")

if os.path.exists(model_save_path):
    print("Loading and evaluating existing model...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    best_val_loss = evaluate(leclair_val)
    print("Initial validation loss: %.4f" % best_val_loss)
else:
    print("No existing model, starting training from scratch")
    best_val_loss = float("inf")

epochs = 15

for epoch in range(1, epochs + 1):
    print("\nStarting epoch %d of %d" % (epoch, epochs))
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(leclair_val)
    print("Finished epoch %d of %d, val loss: %.4f" % (epoch, epochs, val_loss))
    if val_loss < best_val_loss:
        print("Val loss improved from %.4f, saving checkpoint" % best_val_loss)
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
