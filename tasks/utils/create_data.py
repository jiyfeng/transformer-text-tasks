import numpy as np
import os
import csv
import pandas as pd

#path = 'C:/Users/gaurav/Desktop/UVA@second_sem/IS/transformer/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus'
path = '/u/gj3bg/gj3bg/cornell movie-dialogs corpus/'

convlines = []
j = 0
with open(path + 'movie_conversations.txt', encoding="utf8",
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
with open('movie_lines.txt', encoding="utf8", errors='ignore') as m:
    lines = m.readlines()
    for line in lines:
        split = line.split(' +++$+++ ')
        dic[split[0]] = split[-1].split('\n')[0]

print(convlines[:5])
print(list(dic.keys())[:5])

convtexts = []
for tp in convlines:
    t1 = dic[tp[0]]
    t2 = dic[tp[1]]
    if len(t1.split()) > 2:
        convtexts.append([t1, t2])


with open(path + "dialogue_training_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(convtexts[:1000])
    
with open(path + "dialogue_validation_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(convtexts[1000:1300])

print("Saved data into CSV file")
