import numpy as np

convlines = []
j = 0
with open('/Users/mohit/Downloads/cornell movie-dialogs corpus/movie_conversations.txt', encoding="utf8",
          errors='ignore') as f:
    conv = f.readlines()
    # conv.encode('utf-8').strip()
    for c in conv:
        # print(lines)
        split = str(c).split(' +++$+++ ')
        lines = split[-1]
        l = len(lines)
        lines = lines[2:l - 3]
        # print(lines)
        splitlines = lines.split("', '")
        for i in range(len(splitlines) - 1):
            convlines.append((splitlines[i], splitlines[i + 1]))
        # j += 1
        # if j > 3:
        #     exit(0)

dic = {}
with open('/Users/mohit/Downloads/cornell movie-dialogs corpus/movie_lines.txt', encoding="utf8", errors='ignore') as m:
    lines = m.readlines()
    for line in lines:
        # print(line)
        split = line.split(' +++$+++ ')
        dic[split[0]] = split[-1].split('\n')[0]

print(convlines[:5])
print(list(dic.keys())[:5])
# exit(0)
convtexts = []
for tp in convlines:
    # print(tp)
    # if tp[0] in dic and tp[1] in dic:
    t1 = dic[tp[0]]
    t2 = dic[tp[1]]
    if len(t1.split()) > 2:
        convtexts.append((t1, t2))

convtexts = convtexts[5:10]
print(convtexts)
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Encode a text inputs
for i in range(len(convtexts)):
    print("\nconvtext i", i)
    source0 = convtexts[i][0]
    source = " ".join(source0.split()[:3])
    target = convtexts[i][1]
    indexed_tokens = tokenizer.encode(source)

    ## for end of text - <|endoftext|>

    # OnGPU, put everything on cuda
    # tokens_tensor = tokens_tensor.to('cuda')
    # model.to('cuda')
    predicted_texts = list()
    predicted_word = ''
    # Predict all tokens
    count_words = len(indexed_tokens)
    while predicted_word != '<|endoftext|>' and predicted_word != '.' and count_words < 100:
    # while count_words < 50:
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