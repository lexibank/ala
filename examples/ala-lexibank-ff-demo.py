from ala import (
        get_wordlists, FF, affiliate_by_consonant_class, get_asjp, 
        training_data, get_lingpy, get_gb)
from ala import concept2vec, get_db
import numpy as np
from typing import Optional
from tqdm import tqdm
from lingpy.convert.html import colorRange
from collections import defaultdict
from tabulate import tabulate
from scipy.spatial.distance import cosine
from itertools import combinations
from clldutils.misc import slug


db = get_db("lexibank.sqlite3")
converter = concept2vec(db, model="dolgo")

# load asjp to retrieve labels 
grambank = get_gb("grambank.sqlite3")
asjp = get_asjp()
wordlists = {k: v for k, v in get_wordlists("lexibank.sqlite3").items() if k in
             grambank}

train, test = training_data(
        wordlists,
        {k: v[0] for k, v in get_asjp().items()},
        0.8,
        3
        )


fam2idx = {fam: i for i, fam in enumerate(train)}
idx2fam = {v: k for k, v in fam2idx.items()}

# Create training vector
training = [] 

# train: {lng: [data]}
for fam in train:
    # iterate through data for each language
    for lng, data in train[fam].items():
        # set true label
        true_labels = [0 for i in range(len(fam2idx))]
        true_labels[fam2idx[fam]] = 1

        # retrieve data to create converter
        words = [[row[3], row[4].split()] for row in data.values()]

        # convert to long vector of sound classes
        words_as_vector = converter(words)
        training.append([np.array(words_as_vector), np.array(true_labels)])

# Similar vector creation of items in test-set
testing = []
for fam in test:
    # iterate through all items again
    for data in test[fam].values():
        words = [[row[3], row[4].split()] for row in data.values()]
        testing += [[converter(words), fam, fam2idx[fam]]]

nn = FF(
        len(words_as_vector),
        2 * len(fam2idx),  
        len(fam2idx),
        verbose=True
        )
nn.train(training, 20, 0.02)

# create confusion matrix
out = []
confusion = defaultdict(list)
for tst, fam, fam_idx in testing:
    res = nn.predict(tst, nn.input_layer, nn.output_layer)

    # assert that prediction is correct
    if res == fam_idx or res == "?" and fam == "unclassified":
        out += [1]
    else:
        out += [0]
        confusion[fam] += [idx2fam[res]]
print("Correct:", round(sum(out) / len(out), 3))
print("Wrong:", round(out.count(0) / len(out), 3))

# Print confusion table
# table = []
# for a, b in confusion.items():
#     table += [[
#         a, " ".join(["{0}:{1}".format(k, v) for k, v in zip(
#         sorted(set(b), key=lambda x: b.count(x), reverse=True),
#         sorted([b.count(x) for x in sorted(set(b))]))
#                      ])
#         ]]
# print(tabulate(table))


# write distances to phylip file
# weights = nn.output_layer
# tab = []
# for fam, idx in fams.items():
#     cos = cosine(weights[fams[fam]], weights[fams["Sino-Tibetan"]])
#     tab += [[fam, cos]]
# tab = sorted(tab, key=lambda x: x[-1])
# print(tabulate(tab, floatfmt=".2f"))


# distances = [[0.0 for f in fams] for f in fams]
# for (fam_a, idx_a), (fam_b, idx_b) in combinations(fams.items(), r=2):
#     distances[idx_a][idx_b] = distances[idx_b][idx_a] = cosine(weights[idx_a],
#                                                                weights[idx_b])
 
# idx2fam = {v: k for k, v in fams.items()}
# with open("family-distances.tsv", "w") as f:
#     f.write(" "+str(len(fams))+"\n")
#     for i, row in enumerate(distances):
#         f.write(slug(idx2fam[i], lowercase=False)+" ")
#         f.write(" ".join(["{0:.4f}".format(cell) for cell in row])+"\n")
#
#
# # calculate weights (softmax) per language to get the dimensions
# bars, maxis = [], []
# weights_new = []
# for i, row in enumerate(weights):
#     norm = (row-np.min(row))/(np.max(row) - np.min(row))
#     weights_new += [norm/sum(norm)]
# 
# for fam, idx in fams.items():
#     bars += [(fam, weights_new[idx])]
# 
# for i, weight in enumerate(weights_new):
#     maxis += [(i, max([weight[j] for j in range(len(fams))]))]
# maxis = sorted(maxis, key=lambda x: x[1], reverse=True)
# 
# 
# 
# i2i = {j[0]: i for i, j in enumerate(maxis)}
# bars = []
# for fam, idx in fams.items():
#     new_weights = [x for x in maxis]
#     for i, cell in enumerate(weights_new[idx]):
#         new_weights[i2i[i]] = weights_new[idx][i]
#     bars += [(fam, new_weights)]
# 
# bars = sorted(bars, key=lambda x: tuple(x[1]), reverse=True)
# from matplotlib import pyplot as plt
# 
# plt.clf()
# fig = plt.figure(figsize=(20, 10))
# 
# colors = colorRange(len(fams))
# colors[0] = "0.5"
# colors[-1] = "0.5"
# 
# for i, (fam, row) in enumerate(bars):
#     prev = 0
#     vals = []
#     for j, val in enumerate(row):
#         vals += [prev + val]
#         prev = vals[-1]
#     for j, val in enumerate(vals[::-1]):
#         plt.bar(i, val, color=colors[j], align="edge", width=2,
#                 )
# plt.xticks([x+0.5 for x in range(len(bars))], [x[0] for x in bars], rotation=90, fontsize=10)
# plt.savefig("bars.pdf")
