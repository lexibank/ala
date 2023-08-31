"""
Part of this code is based on this tutorial:
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
"""
from collections import defaultdict, Counter
from statistics import mean, stdev
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from ala import get_wl, get_asjp, get_gb, convert_data
from ala import feature2vec, get_db


# Switches for tests - set only one to True!
UTOAZT = False
PANO = False

# Remove (True) or include (False) Isolates/"Unclassified"
ISOLATES = False

RUNS = 100
EPOCHS = 500
BATCH = 1024
HIDDEN = 4  # multiplier for length of fam
LR = 1e-3

# Switch on GPU if available
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("Current device:", device)

scores = []
fam_scores = []
results = defaultdict()  # test cases
fam_confusion = defaultdict()

lb = get_wl("lexibank.sqlite3")
asjp = get_asjp()
wordlists = {k: v for k, v in get_gb("grambank.sqlite3").items()}

# get converter for grambank data
converter = feature2vec(get_db("grambank.sqlite3"))


tacanan = ["esee1248", "arao1248", "cavi1250"]
panoan = ["yawa1260", "poya1241", "ship1254", "yami1256",
          "pano1255", "pano1254", "mats1244", "maru1252",
          "amah1246", "capa1241", "chac1251", "isco1239"]
pano_iso = ["mose1249"]
test_isolates = ["basq1248"]
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
    load="grambank",
    threshold=5)

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
            longdistance_test[lang] = full_data[lang]
        elif lang in pano_iso:
            longdistance_test[lang] = full_data[lang]
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
data = data.to(device)
labels = labels.to(device)
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
        # Linear function
        out = self.fc1(x)

        # Non-linearity
        out = self.ReLU(out)
        # Linear function (readout)
        out = self.fc2(out)

        return out

    def predict(self, vector, language, storage):
        """Predicts based on new data, and stores results in dic."""
        vector = torch.Tensor(np.array([vector[language][2]]))
        vector = vector.to(device)

        outs = model(vector)
        _, prediction = torch.max(outs.data, 1)
        prediction = idx2fam[prediction.item()]
        if lang in storage:
            storage[lang].append(prediction)
        else:
            storage[lang] = [prediction]

        return storage


for run in range(RUNS):
    train_dataset, test_dataset = random_split(tensor_ds, [0.80, 0.20])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH,
                             shuffle=False)

    model = FF(input_dim, hidden_dim, output_dim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ITER = 0
    HIGH = 0
    FAM_HIGH = 0
    BEST = 0
    for epoch in range(EPOCHS):
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
                acc = 100 * CORR / TOTAL
                fam_acc = mean(fam_avg)
                # print(f'Iteration: {ITER}. Loss: {loss.item()}. Average Family Accuracy: {fam_acc}')
                if fam_acc > FAM_HIGH:
                    HIGH = acc
                    BEST = epoch
                    FAM_HIGH = fam_acc
                    # print("Total acc:", acc)
                    # print("Family acc:", fam_acc)
                    torch.save(model.state_dict(), 'best-model-parameters.pt')

    # Summary for best epoch
    scores.append(int(HIGH))
    fam_scores.append(int(FAM_HIGH))
    model.load_state_dict(torch.load('best-model-parameters.pt'))

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
    # print("---------------")

print("---------------")
print("FINAL GRAMBANK:")
for item in results:
    print(item, Counter(results[item]))
for lang in fam_confusion:
    print(lang, ":", fam_confusion[lang])
print(fam2idx)

print("Overall:", round(mean(scores), 2))
print("Standard deviation:", round(stdev(scores), 2))
print("---")
print("Mean family accuracy:", round(mean(fam_scores), 2))
print("Standard deviation:", round(stdev(fam_scores), 2))
