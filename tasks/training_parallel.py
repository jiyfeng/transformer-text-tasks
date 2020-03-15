# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pylab as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torchtext import data
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import logging
logging.basicConfig(level=logging.INFO)


max_len = 40 
batch_size = 64
num_epochs = 10
learning_rate = 0.001
cuda = True
data_dir = '/u/gj3bg/gj3bg/cornell movie-dialogs corpus/'

# Set device type
if cuda and torch.cuda.is_available():
    print("Running on GPU")
    device = torch.device("cuda")
else:
    print("Running on CPU")
    device = torch.device("cpu")

print("-" * 84)
print("Running on device type: {}".format(device))

# os.chdir(data_dir)
convtexts = pd.read_csv(data_dir + 'dialogue_data.csv', sep=',')
convtexts = np.array(convtexts).tolist()
print("Data Example")
print(convtexts[:1])

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


train_file = data_dir + 'dialogue_data.csv'
valid_file  = data_dir + 'dialogue_data.csv'

# Data Parallelism over 4 GPUs
if torch.cuda.device_count() > 1:
    print("Running on ", torch.cuda.device_count(), "GPU's")
    model = nn.DataParallel(model)
    model.to(device)
else:
    model.to(device)

tokenizer.pad_token = '<PAD>'
pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

TEXT = data.Field(use_vocab=False, tokenize=tokenizer.encode, pad_token=pad_index, batch_first =True)
fields = [("src", TEXT), ("trg", TEXT)]

train_data, valid_data = data.TabularDataset.splits(
    path=data_dir,
    train=train_file,
    test=valid_file,
    format="CSV",
    fields=fields,
)

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=batch_size,
    sort_key=lambda x: x.src,
    sort_within_batch=False,
    device=device
)

print("No. of Batches in training data", len(train_iterator))
print("No. of Batches in validation data", len(valid_iterator))

#optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
optimizer = torch.optim.SGD(lr=learning_rate, params=model.parameters(), momentum=0.9)

scheduler = CyclicLR(optimizer, base_lr=learning_rate, max_lr=1, mode="exp_range", gamma=0.9994)

# next(iter(train_iterator))

# for i, batch in enumerate(train_iterator):
#         source = batch.src[2]
#         target = batch.trg[2]
#         print(source)
#         print("Source Given: ", tokenizer.decode(source.tolist()))
#         break

# Start Training
training_loss_list = []
validation_loss_list = []
    
print("-" * 84)
print("Start Training")
for epoch in range(num_epochs):
    model.train()
    training_loss = 0
    for i, batch in enumerate(train_iterator):
        # Get source and target
        print("epoch", epoch, "i", i)
        source = batch.src
        target = batch.trg
        # Trim source text
        if source.size(1) > max_len:
            source = source[:, :max_len]
        if target.size(1) > max_len:
            target = target[:, :max_len]              
        for ind in range(target.shape[1]):
            optimizer.zero_grad()
            label = target[:,ind]
            tokens_tensor = torch.tensor(source)
            out = model(tokens_tensor)
            out = out[0]
            predictions = torch.softmax(out[:, -1, :], dim = 0)
            predictions = torch.log(predictions)
            loss = nn.functional.nll_loss(predictions, torch.tensor(label))
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            predicted_index = torch.argmax(predictions, dim = 1)
            tokens_tensor = torch.cat((tokens_tensor, label.unsqueeze(1)), dim = 1)
    print("epoch", epoch, "training_loss", training_loss)
    training_loss_list.append(training_loss)
    #Evaluation on Validation data
    with torch.no_grad():
        model.eval()
        validation_loss = 0
        for i, batch in enumerate(valid_iterator):
            source = batch.src
            target = batch.trg
            if source.size(1) > max_len:
                source = source[:, :max_len]
            if target.size(1) > max_len:
                target = target[:, :max_len]                  
            for ind in range(target.shape[1]):
                label = target[:,ind]
                tokens_tensor = torch.tensor(source)
                out = model(tokens_tensor)
                out = out[0]
                predictions = torch.softmax(out[:, -1, :], dim = 0)
                predictions = torch.log(predictions)
                loss = nn.functional.nll_loss(predictions, torch.tensor(label))
                validation_loss += loss.item()
                predicted_index = torch.argmax(predictions, dim = 1)
                tokens_tensor = torch.cat((tokens_tensor, predicted_index.unsqueeze(1)), dim = 1)
        print("epoch", epoch, "validation_loss", validation_loss)
        validation_loss_list.append(validation_loss)
        # Print the predicted text
        predicted_text = tokenizer.decode(tokens_tensor[0].tolist())
        print("Target: ", tokenizer.decode(target[0].tolist()))
        print("Source Given: ", tokenizer.decode(source[0].tolist()))
        print("Predicted: ", predicted_text)
        scheduler.step(validation_loss) 
        # Save loss in text file
        c = [training_loss_list, validation_loss_list]
        with open("./loss.txt", "a") as file:
            for x in zip(*c):
                file.write("{0}\t{1}\n".format(*x))      


plt.figure(figsize = (10, 4))
plt.subplot(1, 2, 1)
plt.plot(validation_loss_list, 'bo-', label = 'val-loss')
plt.plot(training_loss_list, 'ro-', label = 'train-loss')
plt.grid('on')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['validation', 'training'], loc='upper right')
plt.savefig("./Loss_vs_epoch.png")

model_file = './saved_model.pkl'
torch.save(model, model_file)
print("training complete")

#Load model for evaluation
# model = torch.load(model_file)
# model.eval()
