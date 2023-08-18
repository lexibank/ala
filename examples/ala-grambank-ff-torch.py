from ala import get_wl, get_asjp, training_data, get_gb
from ala import feature2vec, get_db
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


results = defaultdict()

lb = get_wl("lexibank.sqlite3")
asjp = get_asjp()
wordlists = {k: v for k, v in get_gb("grambank.sqlite3").items()}

# get converter for grambank data
converter = feature2vec(get_db("grambank.sqlite3"))


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


control = []
train, test = training_data(
        wordlists,
        {k: v[0] for k, v in get_asjp().items() if k in lb},
        0.8,
        3
        )

fam2idx = {fam: i for i, fam in enumerate(train)}
idx2fam = {v: k for k, v in fam2idx.items()}
# Create training vector
training_vec = []
train_labels = []

# train: {lng: [data]}
for fam in train:
    # iterate through data for each language
    for lng, data in train[fam].items():
        # set true label
        true_labels = [0 for i in range(len(fam2idx))]
        true_labels[fam2idx[fam]] = 1

        # retrieve data to create converter
        words = [[x[2], x[3]] for x in data.values()]

        # convert to long vector of sound classes
        feature_vector = converter(words)
        training_vec.append(feature_vector)
        train_labels.append(true_labels)

# Similar vector creation of items in test-set
test_vec = []
test_labels = []
for fam in test:
    # iterate through all items again
    for lng, data in test[fam].items():
        true_labels = [0 for i in range(len(fam2idx))]
        true_labels = fam2idx[fam]
        words = [[row[2], row[3]] for row in data.values()]
        feature_vector = converter(words)
        test_vec.append(feature_vector)
        test_labels.append(true_labels)

input_dim = len(feature_vector)
hidden_dim = 4*len(fam2idx)
output_dim = len(fam2idx)

LR = 0.002
EPOCHS = 500
BATCH = 10
ITER = 0

train_data = torch.Tensor(np.array(training_vec))
train_labels = torch.Tensor(np.array(train_labels))
test_data = torch.Tensor(np.array(test_vec))
test_labels = torch.Tensor(np.array(test_labels))


train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

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
    for i, (data, labels) in enumerate(train_loader):
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

        ITER += 1

        if ITER % 500 == 0:
            # Calculate Accuracy
            CORR = 0
            TOTAL = 0
            # Iterate through test dataset
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

            # Print Loss
            print(f'Iteration: {ITER}. Loss: {loss.item()}. Accuracy: {acc}')
