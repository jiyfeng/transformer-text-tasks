# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 02:28:36 2020

@author: gaurav
"""

import numpy as np

convlines = []
with open('movie_conversations.txt') as f:
    conv = f.readlines()
    for c in conv:
        split = c.split(' +++$+++ ')
        lines = split[-1]
        l = len(lines)
        lines = lines[2:l-3]
        splitlines = lines.split("', '")
        for i in range(len(splitlines)-1):
            convlines.append((splitlines[i], splitlines[i+1])) 

dic = {}
with open('movie_lines.txt') as m:
    lines = m.readlines()
    for line in lines:
        split = line.split(' +++$+++ ')
        dic[split[0]] = split[-1].split('\n')[0]

print(convlines[:10])
print(dic)        

convtexts = []
for tp in convlines:
    t1 = dic[tp[0]]
    t2 = dic[tp[1]]
    if len(t1.split()) > 2:
        convtexts.append((t1, t2))
        
convtexts = convtexts[:10]
        
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
text = "Who was Jim Henson ? Jim Henson was a"
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])


# Let’s see how to use GPT2LMHeadModel to generate the next token following our text:

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# If you have a GPU, put everything on cuda
# tokens_tensor = tokens_tensor.to('cuda')
# model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# get the predicted next sub-word (in our case, the word 'man')
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'

print()
print("Text", text)
print("Predicted", predicted_text)