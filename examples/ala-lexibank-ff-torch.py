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
from ala import get_wl, get_asjp, get_gb, convert_data, get_bpt
from ala import concept2vec, get_db


# Switches for tests - set only one to True!
UTOAZT = False
PANO = True

# Remove (True) or include (False) Isolates/"Unclassified"
ISOLATES = True

# Hyperparameters
RUNS = 10
EPOCHS = 500
BATCH = 64
HIDDEN = 3  # multiplier for length of fam
LR = 1e-4

# Switch on GPU if available
if torch.backends.mps.is_available():
    device = "mps"  # MacOS
elif torch.cuda.is_available():
    device = "cuda"  # NVidia
else:
    device = "cpu"

scores = []
results = defaultdict()  # test cases

gb = get_gb("grambank.sqlite3")
asjp = get_asjp()
converter = concept2vec(get_db("lexibank.sqlite3"), model="dolgo")
wordlists = {k: v for k, v in get_wl("lexibank.sqlite3").items() if k in gb}
bpt_wl = {k: v for k, v in get_bpt("bpt.sqlite3").items()}


tacanan = ["esee1248", "taca1256", "arao1248", "cavi1250"]
panoan = ["cash1251", "pano1254", "ship1254", "yami1256", "amah1246",
          "capa1241", "mats1244", "shar1245", "isco1239", "chac1251"]
pano_iso = ["mose1249", "movi1243", "chip1262"]
test_isolates = ["basq1248", "suan1234"]
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
    threshold=7)

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
        else:
            longdistance_test[lang] = bpt_data[lang]
    else:
        full_data[lang] = bpt_data[lang]

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
            pass
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
data = data.to(device)
labels = labels.to(device)
tensor_ds = TensorDataset(data, labels)


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
        vector = vector.to(device)

        outs = model(vector)
        _, prediction = torch.max(outs.data, 1)
        prediction = idx2fam[prediction.item()]
        if lang in storage:
            storage[lang].append(prediction)
        else:
            storage[lang] = [prediction]

        return storage


for i in range(RUNS):
    train_dataset, test_dataset = random_split(tensor_ds, [0.80, 0.20])

    input_dim = data.size()[1]  # Length of data tensor
    hidden_dim = HIDDEN*len(idx2fam)
    output_dim = len(idx2fam)

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
            if ITER % 500 == 0:
                CORR = 0
                TOTAL = 0
                family_results = defaultdict()
                avg_fam = defaultdict()
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

                for fam in family_results:
                    CORR = 0
                    TOTAL = len(family_results[fam])
                    for pred in family_results[fam]:
                        if fam == pred:
                            CORR += 1
                    fam_average = 100 * CORR / TOTAL
                    fam_avg.append(fam_average)

                acc = mean(fam_avg)
                print(f'Iteration: {ITER}. Loss: {loss.item()}. Average Family Accuracy: {acc}')
                if acc > HIGH:
                    HIGH = acc
                    BEST = epoch
                    torch.save(model.state_dict(), 'best-model-parameters.pt')

    # Summary for best epoch
    scores.append(int(HIGH))
    model.load_state_dict(torch.load('best-model-parameters.pt'))
    print("Best epoch:", BEST)
    print("Mean at run", i, ":", round(mean(scores), 2))
    print("---")
    # for lang in family_results:
    #     print(lang, ":", family_results[lang])
    # print(fam2idx)
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

for item in results:
    print(item, Counter(results[item]))
print("---------------")
print("FINAL:")
print("Overall:", round(mean(scores), 2))
print("Standard deviation:", round(stdev(scores), 2))
