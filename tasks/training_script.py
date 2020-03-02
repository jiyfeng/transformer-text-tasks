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
batch_size = 1
num_epochs = 3
learning_rate = 0.01

# Set device type
if cuda and torch.cuda.is_available():
    print("Running on GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("-" * 84)
print("Running on device type: {}".format(device))

data_dir = 'C:/Users/gaurav/Desktop/UVA@second_sem/IS/transformer/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus'
data_dir = '/u/gj3bg/gj3bg/cornell movie-dialogs corpus'

os.chdir(data_dir)
convtexts = pd.read_csv('dialogue_data.csv', sep=',')
convtexts = np.array(convtexts).tolist()
print(convtexts[:3])

train_data_loader = torch.utils.data.DataLoader(convtexts, batch_size=batch_size, shuffle=True)
validation_data_loader = torch.utils.data.DataLoader(convtexts, batch_size=batch_size, shuffle=False)


# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')



# train_file = 'dialogue_data.csv'
# valid_file  = 'dialogue_data.csv'


# def preprocessor(batch):
#     return tokenizer.encode(batch)

# TEXT = data.Field(
#     lower=True,
#     include_lengths=True,
#     batch_first=True,
#     preprocessing=preprocessor,
#     fix_length=max_len
# )


# fields = [("src", TEXT), ("trg", TEXT)]
# train_data, valid_data = data.TabularDataset.splits(
#     path=data_dir,
#     train=train_file,
#     test=valid_file,
#     format="CSV",
#     fields=fields,
# )

# TEXT.build_vocab(train_data, max_size=vocab_size - 2)



# train_iterator, valid_iterator = data.BucketIterator.splits(
#     (train_data, valid_data),
#     batch_size=batch_size,
#     sort_key=lambda x: x.text,
#     sort_within_batch=False,
#     device=device,
# )


optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

# Start Training
print("-" * 84)
print("Start Training")
for epoch in range(num_epochs):
    model.train()
    per_epoch_loss = 0
    for i, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        # Get source and target
        source = list(batch[0])
        target = list(batch[1])
        source = " ".join(source[0].split()[:-1])
        indexed_tokens = tokenizer.encode(source)
        target_index = tokenizer.encode(target)
        for label in target_index:
            tokens_tensor = torch.tensor([indexed_tokens])
            out = model(tokens_tensor)
            out = out[0]
            predictions = torch.softmax(out[0, -1, :], dim = 0)
            predictions = torch.log(predictions)
            loss = nn.functional.nll_loss(predictions.unsqueeze(0), torch.tensor(label).unsqueeze(0))
            per_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            predicted_index = torch.argmax(predictions).item()
            indexed_tokens.append(predicted_index)
        print("epoch", epoch, "i", i)
        predicted_text = tokenizer.decode(indexed_tokens)
        print("Target: ", target)
        print("Source Given: ", source)
        print("Predicted: ", predicted_text)

model_file = 'saved_model.pkl'
torch.save(model, model_file)


# Start Evaluation
print("-" * 84)
print("Start Evaluation")

model = torch.load(model_file)
model.to(device)
model.eval()

# Encode a text inputs
for i in range(len(convtexts)):
    print("\nconvtext i", i)
    source0 = convtexts[i][0]
    source = " ".join(source0.split()[:3])
    target = convtexts[i][1]
    indexed_tokens = tokenizer.encode(source)

    predicted_texts = list()
    predicted_word = ''
    # Predict all tokens
    count_words = len(indexed_tokens)
    while predicted_word != '<|endoftext|>' and predicted_word != '.' and count_words < 100:
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        # get the predicted next sub-word (in our case, the word 'man')
        predicted_index = torch.argmax(predictions[0, -1, :]).item()

        # print("predicted_index", torch.tensor([[predicted_index]]))
        # print("tokens_tensor size", tokens_tensor.size())
        # print("predictions size", predictions.size())

        indexed_tokens.append(predicted_index)
        predicted_word = tokenizer.decode([predicted_index])
        predicted_text = tokenizer.decode(indexed_tokens)

        count_words += 1

    print("Full Source: ", source0)
    print("Source Given: ", source)
    # print("Target: ", target)
    print("Predicted: ", predicted_text)