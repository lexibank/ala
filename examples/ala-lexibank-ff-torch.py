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
from ala import get_wl, get_asjp, get_gb, convert_data
from ala import concept2vec, get_db


EXCLUDE = True
RUNS = 10
LR = 0.002
EPOCHS = 200
BATCH = 10
HIDDEN = 3  # multiplier for length of fam


scores = []
results = defaultdict()  # isolates

gb = get_gb("grambank.sqlite3")
asjp = get_asjp()
converter = concept2vec(get_db("lexibank.sqlite3"), model="dolgo")
# wordlists = {k: v for k, v in get_wl("lexibank.sqlite3").items() if k in gb}
wordlists = {k: v for k, v in get_wl("lexibank.sqlite3").items()}


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
        threshold=2)

    isolates = defaultdict()
    data = []
    labels = []
    idx2fam = defaultdict()
    fam2idx = defaultdict()
    IDX = 0

    for lang in full_data:
        family = full_data[lang][0]
        if EXCLUDE is True:
            if family == "Unclassified":
                isolates[lang] = full_data[lang]
            # somehow, suansu is not in the data! Why?
            elif lang == "suan1234":
                isolates[lang] = full_data[lang]
            else:
                data.append(full_data[lang][2])
                labels.append(full_data[lang][1])
        else:
            data.append(full_data[lang][2])
            labels.append(full_data[lang][1])
        if family not in fam2idx:
            idx2fam[IDX] = family
            fam2idx[family] = IDX
            IDX += 1

    data = torch.Tensor(np.array(data))
    labels = torch.LongTensor(np.array(labels))
    tensor_ds = TensorDataset(data, labels)
    train_dataset, test_dataset = random_split(tensor_ds, [0.8, 0.2])

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

    for epoch in range(EPOCHS):
        for idx, (data, labels) in enumerate(train_loader):
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

    # test Isolates
    for lang in isolates:
        label = isolates[lang][0]
        data = torch.Tensor(np.array([isolates[lang][2]]))

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        predicted = idx2fam[predicted.item()]
        if lang in results:
            results[lang].append(predicted)
        else:
            results[lang] = [predicted]
    print("Mean at run", i, ":", round(mean(scores), 2))

print("---------------")
for item in results:
    print(item, Counter(results[item]))

print("FINAL:")
print("Overall:", round(mean(scores), 2))
print("Standard deviation:", round(stdev(scores), 2))
