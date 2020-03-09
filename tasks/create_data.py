import numpy as np
import os
import csv

# path = 'C:/Users/gaurav/Desktop/UVA@second_sem/IS/transformer/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus'
# path = '/Users/mohit/Downloads/cornell movie-dialogs corpus/'
path = '/u/ms5sw/cornell movie-dialogs corpus/'
if __name__ == '__main__':

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
        if len(t2.split()) > 1:
            convtexts.append([t1, t2])
    print(len(convtexts))
    print(convtexts[:5])

    with open(".data/dialogue_data.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(convtexts)

    print("Saved data into TSV file")
