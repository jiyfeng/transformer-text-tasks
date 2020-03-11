import numpy as np
import os
import csv
import pandas as pd
import torch
import torch.nn as nn
from torchtext import data
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import logging
logging.basicConfig(level=logging.INFO)


max_len = 20 
cuda = False
vocab_size = 40000
batch_size = 16
num_epochs = 2
learning_rate = 0.01

# Set device type
cuda = True
if cuda and torch.cuda.is_available():
    print("Running on GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("-" * 84)
print("Running on device type: {}".format(device))

data_dir = '/u/gj3bg/gj3bg/cornell movie-dialogs corpus'

os.chdir(data_dir)
convtexts = pd.read_csv('dialogue_data.csv', sep=',')
convtexts = np.array(convtexts).tolist()
print(convtexts[:3])

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


train_file = 'dialogue_data.csv'
valid_file  = 'dialogue_data.csv'

if torch.cuda.device_count() > 1:
    print("Running on ", torch.cuda.device_count(), "GPU's")
    model = nn.DataParallel(model)
    model.to(device)
else:
    model.to(device)

def preprocessor(batch):
    return tokenizer.encode(batch)

TEXT = data.Field(
    lower=True,
    include_lengths=True,
    batch_first=True,
    preprocessing=preprocessor,
    fix_length=max_len
)


fields = [("src", TEXT), ("trg", TEXT)]
train_data, valid_data = data.TabularDataset.splits(
    path=data_dir,
    train=train_file,
    test=valid_file,
    format="CSV",
    fields=fields,
)

TEXT.build_vocab(train_data, max_size=vocab_size - 2)


train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=batch_size,
    sort_key=lambda x: x.src,
    sort_within_batch=False,
    device=device,
)

print("train_iterator", train_iterator)

optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())


# Start Training
print("-" * 84)
print("Start Training")
for epoch in range(num_epochs):
    model.train()
    per_epoch_loss = 0
    for i, batch in enumerate(train_iterator):
        optimizer.zero_grad()
        # Get source and target
        #print(batch.src)
        #print(batch.trg)
        print("epoch", epoch, "i", i)
        source = batch.src[0]
        target = batch.trg[0]
        for ind in range(target.shape[1]):
            label = target[:,ind]
            tokens_tensor = torch.tensor(source)
            out = model(tokens_tensor)
            out = out[0]
            predictions = torch.softmax(out[:, -1, :], dim = 0)
            predictions = torch.log(predictions)
            loss = nn.functional.nll_loss(predictions, torch.tensor(label))
            per_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            predicted_index = torch.argmax(predictions).item()
            tokens_tensor = torch.cat((tokens_tensor, label.unsqueeze(1)), dim = 1)
        #predicted_text = tokenizer.decode(indexed_tokens)
        #print("Target: ", target)
        #print("Source Given: ", source)
        #print("Predicted: ", predicted_text)

model_file = 'saved_model.pkl'
torch.save(model, model_file)
print("training complete")

# Start Evaluation
print("-" * 84)
print("Start Evaluation")

# model = torch.load(model_file)
# model.eval()
