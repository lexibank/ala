"""
Part of this code is based on this tutorial:
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
"""
from collections import defaultdict, Counter
from statistics import mean, stdev
import numpy as np
from scipy.spatial import distance
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from ala import get_wl, get_asjp, get_gb, convert_data, get_bpt
from ala import concept2vec, get_db
from clldutils.misc import slug
import csv

# Switches for tests - set only one to True!
UTOAZT = False
PANO = False

# Remove (True) or include (False) Isolates/"Unclassified"
ISOLATES = False

# Hyperparameters
RUNS = 100
EPOCHS = 500
BATCH = 2048
HIDDEN = 4  # multiplier for length of fam
LR = 1e-3

# Switch on GPU if available
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("Current device:", DEVICE)

scores = []
fam_scores = []
list_results = [["Model", "Run", "General", "Family"]]
results = defaultdict()  # test cases

gb = get_gb("grambank.sqlite3")
asjp = get_asjp()
converter = concept2vec(get_db("lexibank.sqlite3"), model="dolgo")
wordlists = {k: v for k, v in get_wl("lexibank.sqlite3").items()}
bpt_wl = {k: v for k, v in get_bpt("bpt.sqlite3").items()}


tacanan = ["esee1248", "taca1256", "arao1248", "cavi1250"]
panoan = ["cash1251", "pano1254", "ship1254", "yami1256", "amah1246",
          "capa1241", "mats1244", "shar1245", "isco1239", "chac1251"]
pano_iso = ["mose1249", "movi1243", "chip1262"]
test_isolates = ["basq1248", "movi1243", "bang1363", "savo1255", "kunz1244", "suan1234"]
northern_uto = ["hopi1249", "utee1244", "sout2969", "cupe1243", "luis1253",
                "cahu1264", "serr1255", "tong1329", "chem1251", "tuba1278",
                "pana1305", "kawa1283", "mono1275", "nort2954", "coma1245"]

# Dictionaries of data to be tested
southern_uto = defaultdict()
longdistance_test = defaultdict()
isolates = defaultdict()

full_data = convert_data(
    wordlists,
    {k: v[0] for k, v in get_asjp().items()},
    converter,
    load="lexibank",
    threshold=5)
bpt_data = convert_data(
    bpt_wl,
    {k: v[0] for k, v in get_asjp().items()},
    converter,
    load="lexibank")

# Split Pano-Tacanan
for lang in bpt_data:
    if PANO is True:
        if lang in panoan:
            full_data[lang] = bpt_data[lang]
            full_data[lang][0] = "Panoan"
        else:
            longdistance_test[lang] = bpt_data[lang]
    else:
        # longdistance_test[lang] = bpt_data[lang]
        pass

data = []
labels = []
idx2fam = defaultdict()
fam2idx = defaultdict()
IDX = 0

for lang in full_data:
    family = full_data[lang][0]
    if family not in fam2idx:
        idx2fam[IDX] = family
        fam2idx[family] = IDX
        IDX += 1

    if PANO is True:
        if lang in tacanan:
            data.append(full_data[lang][2])
            labels.append(fam2idx[family])
        elif lang in panoan:
            family = "Panoan"
            data.append(full_data[lang][2])
            labels.append(fam2idx[family])
        elif lang in pano_iso:
            pass
        elif ISOLATES is True:
            if lang in test_isolates:
                isolates[lang] = full_data[lang]
            elif family == "Unclassified":
                pass
            else:
                data.append(full_data[lang][2])
                labels.append(fam2idx[family])
        else:
            data.append(full_data[lang][2])
            labels.append(fam2idx[family])

    elif UTOAZT is True:
        if family == "Uto-Aztecan" and lang not in northern_uto:
            southern_uto[lang] = full_data[lang]
        elif ISOLATES is True:
            if lang in test_isolates:
                isolates[lang] = full_data[lang]
            elif family == "Unclassified":
                pass
            else:
                data.append(full_data[lang][2])
                labels.append(fam2idx[family])
        else:
            data.append(full_data[lang][2])
            labels.append(fam2idx[family])

    elif ISOLATES is True:
        if lang in test_isolates:
            isolates[lang] = full_data[lang]
        elif family == "Unclassified":
            pass
        else:
            data.append(full_data[lang][2])
            labels.append(fam2idx[family])
    else:
        data.append(full_data[lang][2])
        labels.append(fam2idx[family])

data = torch.Tensor(np.array(data))
labels = torch.LongTensor(np.array(labels))
data = data.to(DEVICE)
labels = labels.to(DEVICE)
tensor_ds = TensorDataset(data, labels)

input_dim = data.size()[1]  # Length of data tensor
hidden_dim = HIDDEN*len(idx2fam)
output_dim = len(idx2fam)


class FF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FF, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.ReLU = nn.ReLU()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ReLU(out)
        out = self.fc2(out)

        return out

    def predict(self, vector, language, storage):
        """Predicts based on new data, and stores results in dic."""
        vector = torch.Tensor(np.array([vector[language][2]]))
        vector = vector.to(DEVICE)

        outs = model(vector)
        _, prediction = torch.max(outs.data, 1)
        prediction = idx2fam[prediction.item()]
        if lang in storage:
            storage[lang].append(prediction)
        else:
            storage[lang] = [prediction]

        return storage


for run in range(RUNS):
    fam_confusion = defaultdict()
    train_dataset, test_dataset = random_split(tensor_ds, [0.80, 0.20])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH,
                             shuffle=False)

    model = FF(input_dim, hidden_dim, output_dim)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ITER = 0
    HIGH = 0
    FAM_HIGH = 0
    BEST = 0
    NO_IMPROVE = 0

    for epoch in range(EPOCHS):
        if NO_IMPROVE < 50:
            for idx, (data, labels) in enumerate(train_loader):
                # Clear gradients
                optimizer.zero_grad()

                # Forward pass to get output/logits
                outputs = model(data)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                # Calculate Accuracy for test set
                ITER += 1
                if ITER % 20 == 0:
                    CORR = 0
                    TOTAL = 0
                    avg_fam = defaultdict()
                    family_results = defaultdict()
                    fam_avg = []
                    for data, labels in test_loader:
                        # Forward pass only to get logits/output
                        outputs = model(data)

                        # Get predictions from the maximum value
                        _, predicted = torch.max(outputs.data, 1)
                        # Labels per family
                        for idx, label in enumerate(labels):
                            pred = int(predicted[idx])
                            label = int(label)
                            if label in family_results:
                                family_results[label].append(pred)
                            else:
                                family_results[label] = [pred]
                    CORR = 0
                    TOTAL = 0
                    for fam in family_results:
                        FAMCORR = 0
                        FAMTOTAL = len(family_results[fam])
                        for pred in family_results[fam]:
                            TOTAL += 1
                            if fam == pred:
                                CORR += 1
                                FAMCORR += 1
                        fam_average = 100 * FAMCORR / FAMTOTAL
                        fam_avg.append(fam_average)
                        fam_confusion[idx2fam[fam]] = fam_average

                    acc = 100 * CORR / TOTAL
                    fam_acc = mean(fam_avg)

                    # print(f'Iteration: {ITER}. Loss: {loss.item()}. Average Family Accuracy: {fam_acc}')
                    if fam_acc > FAM_HIGH:
                        NO_IMPROVE = 0
                        HIGH = acc
                        BEST = epoch
                        FAM_HIGH = fam_acc
                        torch.save(model.state_dict(), 'best-model-parameters.pt')
                    else:
                        NO_IMPROVE += 1
    # Summary for best epoch
    scores.append(int(HIGH))
    fam_scores.append(int(FAM_HIGH))
    model.load_state_dict(torch.load('best-model-parameters.pt'))

    list_results.append([
        "lexibank", run, HIGH, FAM_HIGH
    ])
    # Compute cosine distances for families
    # dist = [[0.0 for f in fam2idx] for f in fam2idx]
    # weights = list(model.parameters())
    # w = weights[2]  # Matrix with length fam2idx
    # for i, v in enumerate(w):
    #     v = v.cpu()
    #     v = v.detach().numpy()
    #     for j, u in enumerate(w):
    #         u = u.cpu()
    #         u = u.detach().numpy()
    #         dist[i][j] = dist[i][j] = distance.cosine(v, u)
    # 
    # with open("family-distances.tsv", "w", encoding="utf8") as f:
    #     # f.write("\t" + "\t".join(list(fam2idx)) + "\n")
    #     f.write(" "+str(len(fam2idx))+"\n")
    #     for i, row in enumerate(dist):
    #         f.write(slug(idx2fam[i], lowercase=False) + " ")
    #         f.write(" ".join(["{0:.4f}".format(cell) for cell in row])+"\n")

    # print("---")
    # print("Best epoch:", BEST)
    # print("Mean at run", run, ":", round(mean(scores), 2))
    # print("---")

    # Long-distance test
    if UTOAZT is True:
        for lang in southern_uto:
            model.predict(southern_uto, lang, results)

    if PANO is True:
        for lang in longdistance_test:
            model.predict(longdistance_test, lang, results)

    if ISOLATES is True:
        for lang in isolates:
            model.predict(isolates, lang, results)

print("---------------")
print("FINAL LEXIBANK:")

for item in results:
    print(item, Counter(results[item]))
# for lang in fam_confusion:
#     print(lang, ":", fam_confusion[lang])
print("Overall accuracy:", round(mean(scores), 2))
print("Standard deviation:", round(stdev(scores), 2))
print("---")
print("Mean family accuracy:", round(mean(fam_scores), 2))
print("Standard deviation:", round(stdev(fam_scores), 2))

with open('lexibank_results.tsv', 'w', encoding="utf8") as csvfile:
    writer = csv.writer(csvfile, delimiter="\t")
    writer.writerows(list_results)
