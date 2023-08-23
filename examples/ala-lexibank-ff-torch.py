"""
Part of this code is based on this tutorial:
- https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
"""
from collections import defaultdict, Counter
from statistics import mean, stdev
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from ala import get_wl, get_asjp, get_gb, convert_data, get_bpt
from ala import concept2vec, get_db


EXCLUDE = True
RUNS = 10
LR = 0.002
EPOCHS = 200
BATCH = 10
HIDDEN = 3  # multiplier for length of fam


scores = []
results = defaultdict()  # test cases

gb = get_gb("grambank.sqlite3")
asjp = get_asjp()
converter = concept2vec(get_db("lexibank.sqlite3"), model="dolgo")
# wordlists = {k: v for k, v in get_wl("lexibank.sqlite3").items() if k in gb}
wordlists = {k: v for k, v in get_wl("lexibank.sqlite3").items()}
bpt_wl = {k: v for k, v in get_bpt("bpt.sqlite3").items()}

tacanan = ["esee1248", "taca1256", "arao1248", "cavi1250"]
panoan =  ["cash1251", "pano1254", "ship1254", "yami1256", "amah1246",
            "capa1241", "mats1244", "shar1245", "isco1239", "chac1251"]
isolates = ["mose1249", "movi1243", "chip1262"]


class FF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FF, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.tanh = nn.Tanh()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function
        out = self.fc1(x)

        # Non-linearity
        out = self.tanh(out)
        # Linear function (readout)
        out = self.fc2(out)

        return out


for i in range(RUNS):
    full_data = convert_data(
        wordlists,
        {k: v[0] for k, v in get_asjp().items()},
        converter,
        load="lexibank",
        threshold=3)

    converter = concept2vec(get_db("lexibank.sqlite3"), model="dolgo")
    bpt_data = convert_data(
        bpt_wl,
        {k: v[0] for k, v in get_asjp().items()},
        converter,
        load="lexibank",
        threshold=3)

    reduced_data = defaultdict()
    for lang in full_data:
        if lang in isolates:
            pass
        elif lang in panoan:
            pass
        elif lang in tacanan:
            pass
        else:
            reduced_data[lang] = full_data[lang]

    longdistance_test = defaultdict()
    for lang in bpt_data:
        if lang in panoan:
            reduced_data[lang] = bpt_data[lang]
        else:
            longdistance_test[lang] = bpt_data[lang]

    isolates = defaultdict()
    data = []
    labels = []
    idx2fam = defaultdict()
    fam2idx = defaultdict()
    IDX = 0

    for lang in reduced_data:
        family = reduced_data[lang][0]
        if family not in fam2idx:
            idx2fam[IDX] = family
            fam2idx[family] = IDX
            IDX += 1
        if EXCLUDE is True:
            if family == "Unclassified":
                isolates[lang] = reduced_data[lang]
            # somehow, suansu is not in the data! Why?
            elif lang == "suan1234":
                isolates[lang] = reduced_data[lang]
            else:
                data.append(reduced_data[lang][2])
                labels.append(fam2idx[family])
        else:
            data.append(reduced_data[lang][2])
            labels.append(fam2idx[family])


    data = torch.Tensor(np.array(data))
    labels = torch.LongTensor(np.array(labels))
    tensor_ds = TensorDataset(data, labels)
    train_dataset, test_dataset = random_split(tensor_ds, [0.99, 0.01])

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    ITER = 0
    for epoch in range(EPOCHS):
        for idx, (data, labels) in enumerate(train_loader):
            # print(ITER, idx)
            # Clear gradients w.r.t. parameters
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
            for data, labels in test_loader:
                # Forward pass only to get logits/output
                outputs = model(data)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                # Total number of labels
                TOTAL += labels.size(0)
                # Total correct predictions
                CORR += (predicted == labels).sum()

            acc = 100 * CORR / TOTAL
            scores.append(int(acc))
            print(f'Iteration: {ITER}. Loss: {loss.item()}. Accuracy: {acc}')

    # test Isolates or LongDistance
    for lang in longdistance_test:
        data = torch.Tensor(np.array([longdistance_test[lang][2]]))

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        predicted = idx2fam[predicted.item()]
        if lang in results:
            results[lang].append(predicted)
        else:
            results[lang] = [predicted]
    # print("Mean at run", i, ":", round(mean(scores), 2))

print("---------------")
for item in results:
    print(item, Counter(results[item]))

# print("FINAL:")
# print("Overall:", round(mean(scores), 2))
# print("Standard deviation:", round(stdev(scores), 2))
