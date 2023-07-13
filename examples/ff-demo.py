from ala import get_wordlists, FF, affiliate_by_consonant_class, get_asjp, training_data, get_lingpy
from ala import concept2vec, get_db
import numpy as np
from typing import Optional
from tqdm import tqdm
from lingpy.convert.html import colorRange
from collections import defaultdict
from tabulate import tabulate


# get database and converter
db = get_db("lexibank.sqlite3")
converter = concept2vec(db, model="dolgo")


asjp = get_asjp()
wordlists = get_wordlists("lexibank.sqlite3")

train, test = training_data(
        wordlists, 
        {k: v[0] for k, v in get_asjp().items()}, 
        0.8,
        3
        )

cognates = {}
cogid = 0
training = []
training2 = []
fams = {fam: i for i, fam in enumerate(train)}
for fam in train:
    for lng, data in train[fam].items():
        for idx, row in data.items():
            this_cogid = cognates.get(row[-1])
            if this_cogid is None:
                cognates[row[-1]] = cogid
                cogid += 1

for fam in train:
    for lng, data in train[fam].items():
        cogs = [cognates[row[-1]] for row in data.values() if row[-1] in cognates]
        vec = [0 for i in range(len(cognates))]
        for c in cogs:
            vec[c] = 1
        vec2 = [0 for i in range(len(fams))]
        vec2[fams[fam]] = 1
        training += [[vec, vec2]]

        words = [[row[3], row[4].split()] for row in data.values()]
        vec3 = converter(words)

        training2 += [[vec3, vec2]]



testing, testing_true = [], []
testing2 = []
for fam in test:
    for lng, data in test[fam].items():
        cogs = [cognates[row[-1]] for row in data.values() if row[-1] in cognates]
        vec = [0 for i in range(len(cognates))]
        for c in cogs:
            vec[c] = 1
        testing += [vec]
        testing_true += [[fam, fams[fam]]]

        words = [[row[3], row[4].split()] for row in data.values()]
        testing2 += [converter(words)]

idx2fam = {v: k for k, v in fams.items()}


nn = FF(
        len(vec3),
        2 * len(fams),
        len(fams),
        verbose=True
        )
nn.train(training2, 20, 0.02)

out = []
confusion = defaultdict(list)
for tst, trueres in zip(testing2, testing_true):
    res = nn.predict(tst, nn.input_layer, nn.output_layer)
    if len(res) == 1:
        if res[0] == trueres[-1]:
            out += [1]
        else:
            out += [0]
            confusion[trueres[0]] += [idx2fam[res[0]]]
    else:
        out += [0]
        confusion[trueres[0]] += ["?"]
print(sum(out) / len(out), out.count(0) / len(out))

table = []
for a, b in confusion.items():
    table += [[
        a, " ".join(["{0}:{1}".format(k, v) for k, v in zip(
        sorted(set(b), key=lambda x: b.count(x), reverse=True),
        sorted([b.count(x) for x in sorted(set(b))]))
                     ])
        ]]
print(tabulate(table))



# print("Compiled Training Data")
# print("Vector-Length is {0}".format(len(vec3)))
# epoch_loss, weights_in, weights_out = train_ff(
#         len(vec3), #len(cognates),
#         len(fams), #int(3 * len(fams) + 0.5),
#         len(fams),
#         20,
#         np.array(training2, dtype="longdouble"),
#         0.02,
#         verbose=True)
# 
# 
# from tabulate import tabulate
# from scipy.spatial.distance import cosine
# 
# weights = np.transpose(weights_out[-1])
# tab = []
# for fam, idx in fams.items():
#     cos = cosine(weights[fams[fam]], weights[fams["Sino-Tibetan"]])
#     tab += [[fam, cos]]
# tab = sorted(tab, key=lambda x: x[-1])
# print(tabulate(tab, floatfmt=".2f"))
# 
# from itertools import combinations
# from clldutils.misc import slug
# 
# distances = [[0.0 for f in fams] for f in fams]
# for (fam_a, idx_a), (fam_b, idx_b) in combinations(fams.items(), r=2):
#     distances[idx_a][idx_b] = distances[idx_b][idx_a] = cosine(weights[idx_a],
#                                                                weights[idx_b])
# 
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
