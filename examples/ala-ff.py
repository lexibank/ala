from ala import get_wordlists, affiliate_by_consonant_class, get_asjp, training_data, get_lingpy
from ala import concept2vec, get_db
import numpy as np
from typing import Optional
from tqdm import tqdm

def forward(hidden_in, hidden_out, tvecs):
    
    hidden_layer = np.dot(hidden_in.T, tvecs)
    u = np.dot(hidden_out.T, hidden_layer)
    y_predicted = softmax(u)

    return y_predicted, hidden_layer, u


def get_error(y, context):
    
    idxs = set([i for i in range(len(context)) if context[i] == 1])
    idxs_l = len(idxs)
        
    total_error = [
            (p - 1) + (idxs_l - 1) * p if i in idxs else idxs_l * p for i, p in enumerate(y)
            ]            
    return  np.array(total_error)


def get_loss(uvecs, context):
    if [x for x in uvecs if x > 700]:
        for i in range(len(uvecs)):
            if uvecs[i] > 700:
                uvecs[i] = 700

    sum_1 = -1 * sum(
            [uvecs[i] for i, c in enumerate(context) if c == 1]) 
    sum_2 = sum(context) * np.log(np.sum(np.exp(uvecs)))
    return sum_1 + sum_2        

def backward(
        hidden_in, hidden_out, total_error, hidden_layer, tvecs,
        learning_rate):
    dl_hidden_in = np.outer(tvecs, np.dot(hidden_out, total_error.T))
    dl_hidden_out = np.outer(hidden_layer, total_error)

    hidden_in = hidden_in - (learning_rate * dl_hidden_in)
    hidden_out = hidden_out - (learning_rate * dl_hidden_out)

    return hidden_in, hidden_out



def train_ff(
        cognates: int,
        shibbolets: int,
        families: int,
        epochs: int,
        training_data: Optional[list[list, list]],
        learning_rate: float,
        verbose: bool=False,
        interval: int=1
        ):

    hidden_in = np.random.uniform(-1, 1, (cognates, shibbolets))
    hidden_out = np.random.uniform(-1, 1, (shibbolets, families))
    
    
    #For analysis purposes
    epoch_loss = []
    weights_in = []
    weights_out = []
    
    for epoch in range(epochs):
        loss = 0
        for target, context in tqdm(training_data, desc="epoch {0}".format(epoch)):
            y, hidden, uvecs = forward(
                    hidden_in,
                    hidden_out,
                    target
                    )

            total_error = get_error(y, context)

            hidden_in, hidden_out = backward(
                    hidden_in,
                    hidden_out,
                    total_error, 
                    hidden,
                    target,
                    learning_rate)
            
            loss_temp = get_loss(uvecs, context)
            loss += loss_temp
        
        epoch_loss.append(loss)
        weights_in.append(hidden_in)
        weights_out.append(hidden_out)
        
        if verbose:
            if epoch % interval == 0:
                print('Epoch: {0} Loss: {1}'.format(epoch, loss))

    return epoch_loss, np.array(weights_in), np.array(weights_out)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def predict(x, weights_in, weights_out):
    
    y, hidden, u = forward(weights_in, weights_out, x)

    return [i for i, v in enumerate(y) if v >= 0.99]
    

# get database and converter
db = get_db("lexibank.sqlite3")
converter = concept2vec(db)


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






print("Compiled Training Data")
print("Vector-Length is {0}".format(len(vec3)))
epoch_loss, weights_in, weights_out = train_ff(
        len(vec3), #len(cognates),
        int(1.0 * len(fams) + 0.5),
        len(fams),
        20,
        np.array(training2, dtype="object"),
        0.05,
        verbose=True)


from tabulate import tabulate
from scipy.spatial.distance import cosine

weights = np.transpose(weights_out[-1])
tab = []
for fam, idx in fams.items():
    cos = cosine(weights[fams[fam]], weights[fams["Sino-Tibetan"]])
    tab += [[fam, cos]]
tab = sorted(tab, key=lambda x: x[-1])
print(tabulate(tab, floatfmt=".2f"))

from itertools import combinations
from clldutils.misc import slug

distances = [[0.0 for f in fams] for f in fams]
for (fam_a, idx_a), (fam_b, idx_b) in combinations(fams.items(), r=2):
    distances[idx_a][idx_b] = distances[idx_b][idx_a] = cosine(weights[idx_a],
                                                               weights[idx_b])

idx2fam = {v: k for k, v in fams.items()}
with open("family-distances.tsv", "w") as f:
    f.write(" "+str(len(fams))+"\n")
    for i, row in enumerate(distances):
        f.write(slug(idx2fam[i], lowercase=False)+" ")
        f.write(" ".join(["{0:.4f}".format(cell) for cell in row])+"\n")

good, bad = [], []
for tst, trueres in zip(testing2, testing_true):
    res = predict(tst, weights_in[-1], weights_out[-1])
    if len(res) == 1:
        if res[0] == trueres[-1]:
            good += [1]
        else:
            bad += [1]
    else:
        if not res:
            bad += [1]
        else:
            bad += [1]
print(sum(good) / len(testing), sum(bad) / len(testing))
    
