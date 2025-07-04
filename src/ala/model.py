import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict


class FF(nn.Module):
    """Network model with functions for forward-pass and predictions."""
    def __init__(self, database_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(database_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward-pass with two hidden layers."""
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc_out(out)

        return out

    def predict(self, vector, language, idx2fam, device):
        """Predicts based on new data."""
        vector = torch.Tensor(np.array([vector[language][2]]))
        vector = vector.to(device)

        outs = self(vector)
        _, prediction = torch.max(outs.data, 1)
        prediction = idx2fam[prediction.item()]

        return prediction


def train(data, runs, epochs=5000, batch=2048, hidden=4, lr=1e-3, test_langs=None, experiment=False):
    """Training the model."""
    result_per_fam = defaultdict(list)
    results_experiment = defaultdict(list)

    # Switch on GPU if available
    device = 'mps' if torch.backends.mps.is_available() \
        else 'cuda' if torch.cuda.is_available() \
        else 'cpu'
    print('Current device:', device)

    idx2fam = dict(enumerate(set((data[lang][0] for lang in data))))
    fam2idx = {family: idx for idx, family in enumerate(set(data[lang][0] for lang in data))}

    tests = {lang: data[lang] for lang in data if lang in test_langs}
    features = [data[lang][2] for lang in data if lang not in test_langs]
    labels = [fam2idx[data[lang][0]] for lang in data if lang not in test_langs]

    # Summary stats
    summary_stats = {
        'Number of families': len(fam2idx),
        'Number of languages': len(data),
        'Size of vector': len(features[0])
    }
    print(summary_stats)

    # Weights for CrossEntropy
    fam2w = defaultdict(int)
    for family in [data[lang][0] for lang in data]:
        fam2w[family] += 1

    class_weights = [round(fam2w[max(fam2w, key=fam2w.get)] / fam2w[fam], 3) for fam in fam2w]
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Data to tensor
    features = torch.Tensor(np.array(features))
    all_labels = torch.LongTensor(np.array(labels))
    tensor_ds = TensorDataset(features, all_labels)

    for run in range(runs):
        print('--- New Run: ', run+1, '/', runs, '---')
        fam_final = defaultdict()
        train_ds, test_ds = train_test_split(
            tensor_ds, test_size=0.2, stratify=all_labels
            )

        train_ds = [(item[0].to(device), item[1].to(device)) for item in train_ds]
        test_ds = [(item[0].to(device), item[1].to(device)) for item in test_ds]

        train_loader = DataLoader(dataset=train_ds, batch_size=batch, shuffle=True)
        test_loader = DataLoader(dataset=test_ds, batch_size=batch)

        model = FF(features.size()[1], hidden*len(idx2fam), len(idx2fam)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        no_improve = 0
        best_macro = 0
        iters = 0

        for _ in range(epochs):
            if no_improve < 20:
                for idx, (train_data, labels) in enumerate(train_loader):
                    optimizer.zero_grad()   # Clear gradients
                    outputs = model(train_data)   # Forward pass to get output/logits
                    loss = criterion(outputs, labels)
                    loss.backward()         # Getting gradients w.r.t. parameters
                    optimizer.step()        # Updating parameters

                    iters += 1
                    if iters % 100 == 0:
                        family_results = defaultdict()
                        fam_avg = defaultdict()
                        for test_data, labels in test_loader:
                            outputs = model(test_data)
                            _, preds = torch.max(outputs.data, 1)

                            # Labels per family
                            for idx, label in enumerate(labels):
                                family_results.setdefault(int(label), []).append(int(preds[idx]))

                        for fam in family_results:
                            corr = sum(1 for pred in family_results[fam] if fam == pred)
                            total = len(family_results[fam])
                            fam_avg[idx2fam[fam]] = [100 * corr / total, total]

                        bal_acc = sum([fam_avg[fam][0] for fam in fam_avg]) / len(fam_avg)
                        print(f'Iter: {iters}. Loss: {round(loss.item(), 5)}. Bal. acc.: {bal_acc}.')

                        if bal_acc > best_macro:
                            best_macro = bal_acc
                            fam_final = fam_avg
                            no_improve = 0
                            torch.save(model.state_dict(), 'results/best-mpar.pt')
                        else:
                            no_improve += 1

        model.load_state_dict(torch.load('results/best-mpar.pt', weights_only=True))

        if experiment:
            for lang in tests:
                results_experiment[lang, tests[lang][0]].append(
                    model.predict(tests, lang, idx2fam, device))

        for fam, res in fam_final.items():
            result_per_fam[fam].append(
                [run + 1, fam, fam2w[fam], res[1], round(res[0], 3)]
                )

        result_per_fam['TOTAL'].append(
                [run + 1, 'TOTAL', len(data), len(test_ds), best_macro])

    return result_per_fam, results_experiment
