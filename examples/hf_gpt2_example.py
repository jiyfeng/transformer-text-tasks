import torch
from torch.nn import functional as F

from tasks.utils.decoder import top_p_sampling

convlines = []
j = 0
with open('/Users/mohit/Downloads/cornell movie-dialogs corpus/movie_conversations.txt', encoding="utf8",
          errors='ignore') as f:
    conv = f.readlines()
    for c in conv:
        split = str(c).split(' +++$+++ ')
        lines = split[-1]
        l = len(lines)
        lines = lines[2:l - 3]
        splitlines = lines.split("', '")
        for i in range(len(splitlines) - 1):
            convlines.append((splitlines[i], splitlines[i + 1]))

dic = {}
with open('/Users/mohit/Downloads/cornell movie-dialogs corpus/movie_lines.txt', encoding="utf8", errors='ignore') as m:
    lines = m.readlines()
    for line in lines:
        split = line.split(' +++$+++ ')
        dic[split[0]] = split[-1].split('\n')[0]


convtexts = []
for tp in convlines:
    # print(tp)
    # if tp[0] in dic and tp[1] in dic:
    t1 = dic[tp[0]]
    t2 = dic[tp[1]]
    if len(t1.split()) > 2:
        convtexts.append((t1, t2))

# convtexts = convtexts[5:10]

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
    full_source = convtexts[i][0]
    source = " ".join(full_source.split()[:2])
    # target = convtexts[i][1]

    ## for end of text - <|endoftext|>
    # OnGPU, put everything on cuda
    # tokens_tensor = tokens_tensor.to('cuda')
    # model.to('cuda')

    k = 50
    p = 0.8
    num_words = 20
    temperature = 1

    ###
    is_first = True
    indexed_tokens = tokenizer.encode(source)
    i = 0
    while i < num_words:
        last_word = tokenizer.decode(indexed_tokens[-1])
        if not is_first and last_word == ".":
            is_first = False
        # if last_word == '"':
        #     continue
        if last_word == "<|endoftext|>":
            break

        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0][0][-1]

        probabilities = F.softmax(predictions / temperature)

        # predicted_index = top_k_sampling(probabilities, k=k)
        predicted_index = top_p_sampling(probabilities, p=p)

        indexed_tokens.append(predicted_index)
        i += 1
    predicted_text = tokenizer.decode(indexed_tokens)

    print("Full Source: ", full_source)
    print("Source Given: ", source)
    print("Predicted: ", predicted_text)
