from ala import get_wordlists, FF, affiliate_by_consonant_class, get_asjp, training_data, get_gb, get_gb_new
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


# Family/Genus in ASJP
LEVEL = 0
# split percentage
SPLIT = 0.8
# mininum number of languages in family
MIN_LANGS = 3

# number of epochs
EPOCHS = 100
# learning rate
# cut down to 0.01 or less to avoid vanishing gradient!
LR = 0.01

# get database and converter
db = get_db("lexibank.sqlite3")
# converts data into vector
converter = concept2vec(db, model="dolgo")

# loads asjp and wordlists
asjp = get_asjp()
wordlists = get_gb_new("grambank.sqlite3")

# splits data
train, test = training_data(
        wordlists,
        {k: v[LEVEL] for k, v in get_asjp().items()},
        SPLIT,
        MIN_LANGS
        )

# create dictionary for all parameters
params = {}
answers = defaultdict()
P_ID = 0
# creates numeric ID's for language families in data
fams = {fam: i for i, fam in enumerate(train)}
# families in training data
for fam in train:
    # languages and families in training data
    for lng, data in train[fam].items():
        # idx and row of items of language
        for idx, row in data.items():
            # retrieve value of dictionary for concept
            this_P_ID = params.get(row[-1])
            # if not in dic, create new entry
            if this_P_ID is None:
                params[row[-1]] = P_ID
                answers[row[-1]] = [row[2]]
                P_ID += 1
            elif row[2] not in answers[row[-1]]:
                answers[row[-1]].append(row[2])
# !!! create vector with list of all parameters that can be set to 1/0
# print(answers)
for answer in answers:
    if 0 not in answers[answer]:
        answers[answer].append(0)
# Create training vector
training = []

# train: {lng: [data]}
for fam in train:
    # iterate through data for each language
    for lng, data in train[fam].items():
        vec = [0 for i in range(len(params))]
        for idx, item in enumerate(answers):
            # max value is 3, so range needs to be 4
            # all vectors of same length
            vec[idx] = [0 for i in range(4)]
        # set value at ID of parameter as value of parameter in data
        # Note: This 0's all missing data
        for row in data.values():
            old_entry = vec[params[row[-1]]]
            vec[params[row[-1]]][int(row[2])] = 1
            # print("---")
            # print("param:", row[-1])
            # print("new value:", vec[params[row[-1]]])
            # print("value:", row[2])
        # parallel to full-length vector for concepts, make this for families
        vec2 = [0 for i in range(len(fams))]
        # one-hot encode the family that is looped
        vec2[fams[fam]] = 1
        vec = np.array(vec)
        vec = vec.reshape(-1)  # flat array
        vec2 = np.array(vec2)
        # vec2 = vec2.reshape(1, len(vec2))
        # print(vec)
        # print(vec.shape)
        # print(vec2.shape)
        # print("--")
        # attach sound class vector and family vector to training2
        training += [[vec, vec2]]
        # +++ GB equivalent: binary vector for parameters
        # +++ Try out for GB: one-hot encoding

# Similar vector creation of items in test-set
# +++ modify this to a function call instead
testing, testing_true = [], []
for fam in test:
    # iterate through all items again
    for lng, data in test[fam].items():
        vec = [0 for i in range(len(params))]
        for idx, item in enumerate(answers):
            # max value is 3, so range needs to be 4
            # all vectors of same length
            vec[idx] = [0 for i in range(4)]
        # set value at ID of parameter as value of parameter in data
        # Note: This 0's all missing data
        for row in data.values():
            old_entry = vec[params[row[-1]]]
            vec[params[row[-1]]][int(row[2])] = 1

        vec = np.array(vec)
        vec = vec.reshape(-1)
        testing += [vec]
        testing_true += [[fam, fams[fam]]]

# idx2fam:  dict: {ID: "Name"}
# fams:     dict: {"Name": ID}
idx2fam = {v: k for k, v in fams.items()}


nn = FF(
        len(vec),      # input: length of vec3
        2 * len(fams),  # hidden: vec with double the length of fams
        len(fams),      # output: vec with length of fams
        verbose=True
        )
# input sound-class and family vector, epochs, and LR
nn.train(training, EPOCHS, LR)

# create confusion matrix
out = []
confusion = defaultdict(list)
for tst, trueres in zip(testing, testing_true):
    # print(tst)
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
