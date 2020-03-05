import numpy as np
import os
import csv

#path = 'C:/Users/gaurav/Desktop/UVA@second_sem/IS/transformer/tf/'
path = '/Users/mohit/Downloads/cornell movie-dialogs corpus/'

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
with open(path + 'movie_lines.txt', encoding="utf8", errors='ignore') as m:
    lines = m.readlines()
    for line in lines:
        split = line.split(' +++$+++ ')
        dic[split[0]] = split[-1].split('\n')[0]

convtexts = []
for tp in convlines:
    t1 = dic[tp[0]]
    t2 = dic[tp[1]]
    if len(t1.split()) > 2:
        convtexts.append([t1, t2])


with open(".data/dialogue_data.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(convtexts[:500])

print("Saved data into TSV file")