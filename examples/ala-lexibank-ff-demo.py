from ala import get_wordlists, FF, affiliate_by_consonant_class, get_asjp, training_data, get_lingpy
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


# split percentage
SPLIT = 0.8
# mininum number of languages in family
MIN_LANGS = 3

# number of epochs
EPOCHS = 10
# learning rate
# !!! cut down to 0.01 or less to avoid vanishing gradient!
LR = 0.01

# get database and converter
db = get_db("lexibank.sqlite3")
# converts data into vector
converter = concept2vec(db, model="dolgo")

# loads asjp and wordlists
asjp = get_asjp()
wordlists = get_wordlists("lexibank.sqlite3")

# splits data
train, test = training_data(
        wordlists,
        {k: v[0] for k, v in get_asjp().items()},
        SPLIT,
        MIN_LANGS
        )

# create dictionary for all concepts
# ??? are we talking concepts, or cognates?
cognates = {}
COGID = 0
# creates numeric ID's for language families in data
fams = {fam: i for i, fam in enumerate(train)}
# families in training data
for fam in train:
    # languages and families in training data
    for lng, data in train[fam].items():
        # idx and row of items of language
        for idx, row in data.items():
            # retrieve value of dictionary for concept
            this_cogid = cognates.get(row[-1])
            # if not in dic, create new entry
            if this_cogid is None:
                cognates[row[-1]] = COGID
                COGID += 1

# Create training vector
# ??? Why training and training2?
training = []  # present concepts and family
training2 = []  # sound class vector and family

# train: {lng: [data]}
for fam in train:
    # iterate through data for each language
    for lng, data in train[fam].items():
        # list of cogs for language
        # ??? Why is the if-condition necessary?
        # ??? Are not all concepts added to this list?
        # previously:  if row[-1] in cognates
        cogs = [cognates[row[-1]] for row in data.values()]

        # vector of 0's for whole set of concepts
        vec = [0 for i in range(len(cognates))]
        # set 1 all concepts that are present in data
        for c in cogs:
            vec[c] = 1
        # parallel to full-length vector for concepts, make this for families
        vec2 = [0 for i in range(len(fams))]
        # one-hot encode the family that is looped
        vec2[fams[fam]] = 1

        # add concepts and one-hot family vector to training data
        training += [[vec, vec2]]

        # words: [concept, [form] ]
        words = [[row[3], row[4].split()] for row in data.values()]
        # convert to long vector of sound classes
        vec3 = converter(words)
        vec3 = np.array(vec3)
        vec2 = np.array(vec2)
        # print(vec3.shape)
        # print(vec2.shape)
        # attach sound class vector and family vector to training2
        training2 += [[vec3, vec2]]


# Similar vector creation of items in test-set
# +++ modify this to a function call instead
testing, testing_true = [], []
testing2 = []
for fam in test:
    # iterate through all items again
    for lng, data in test[fam].items():
        cogs = [cognates[row[-1]] for row in data.values() if row[-1] in cognates]
        vec = [0 for i in range(len(cognates))]
        for c in cogs:
            vec[c] = 1
        testing += [vec]
        testing_true += [[fam, fams[fam]]]

        words = [[row[3], row[4].split()] for row in data.values()]
        testing2 += [converter(words)]

# idx2fam:  dict: {ID: "Name"}
# fams:     dict: {"Name": ID}
idx2fam = {v: k for k, v in fams.items()}

nn = FF(
        len(vec3),      # input: length of vec3
        2 * len(fams),  # hidden: vec with double the length of fams
        len(fams),      # output: vec with length of fams
        verbose=True
        )
# input sound-class and family vector, epochs, and LR
nn.train(training2, EPOCHS, LR)

# create confusion matrix
out = []
confusion = defaultdict(list)
for tst, trueres in zip(testing2, testing_true):
    # Run prediction
    res = nn.predict(tst, nn.input_layer, nn.output_layer)
    # check if prediction has been made
    if len(res) == 1:
        # assert that prediction is correct
        if res[0] == trueres[-1]:
            out += [1]
        else:
            out += [0]
            confusion[trueres[0]] += [idx2fam[res[0]]]
            # print(trueres[0])
    # else condition: No classification is made
    else:
        out += [0]
        # add item to confustion matrix
        confusion[trueres[0]] += ["?"]
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
