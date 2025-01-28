import argparse
import csv
import sys
from collections import defaultdict, Counter
from statistics import mean, stdev
from tabulate import tabulate
import numpy as np
from scipy.spatial import distance
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from ala import get_lb, get_asjp, get_gb, convert_data, get_other
from ala import concept2vec, feature2vec, get_db, extract_branch
from clldutils.misc import slug
from torchmetrics import F1Score


# Hyperparameters
EPOCHS = 200
BATCH = 2096
HIDDEN = 4  # multiplier for length of fam
LEARNING_RATE = 1e-3
MIN_LANGS = 5


def run_ala(input, test_isolates=False, test_longdistance=False, distances=False, runs=100):
    """Defines the workflow for data loading in the different settings."""
    # Check number of runs
    if runs < 10:
        print(f'The current number of runs is {runs}. We recommend a value higher than 10 to\
              ensure that all language families are well represented in the results.')

    # Load data for testing
    tests = defaultdict()
    northern_uto = extract_branch(gcode='nort2953')
    anatolian = extract_branch(gcode='anat1257')
    tocharian = extract_branch(gcode='tokh1241')
    sinitic = extract_branch(gcode='sini1245')

    isolates = ['bang1363', 'basq1248', 'mapu1245', 'kusu1250', 'cara1273']

    # Switch on GPU if available
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Current device:', device)

    # Empty lists and dicts for results
    table = []
    fam_scores = []
    f1_scores = []
    results_per_fam = defaultdict()  # store family results
    results = defaultdict()  # store experiment results

    # Setup for databases
    asjp = get_asjp()
    grambank = get_gb()
    lexibank = get_lb()
    gb_conv, gb_keys = feature2vec(get_db('data/grambank.sqlite3'))
    lb_conv, lb_keys = concept2vec(get_db('data/lexibank.sqlite3'), model='dolgo')
    load = 'grambank' if input == 'grambank' else 'lexical'
    converter = gb_conv if input == 'grambank' else lb_conv

    # Load main data
    data_map = {
        'lexibank': dict(lexibank.items()),
        'lb_mod': dict(lexibank.items()),
        'combined': dict(lexibank.items()),
        'grambank': dict(grambank.items()),
        'asjp': dict(get_other(mode='asjp').items())
    }

    if input not in data_map:
        print("Invalid data selection. Please choose 'lexibank', 'grambank', 'combined' or 'asjp'")
        sys.exit()

    wordlists = data_map[input]
    n_pars = gb_keys if input == 'grambank' else gb_keys + lb_keys if input == 'combined' else lb_keys

    if input != 'combined':
        data = convert_data(
            wordlists,
            {k: v[0] for k, v in asjp.items()},
            converter,
            load=load,
            threshold=MIN_LANGS
            )

    # test integration of ASJP Genus
    # test = {k: v[1] for k, v in get_asjp().items()}
    # print(test)

    # Set up combination of LB and GB
    elif input == 'combined':
        data = convert_data(
            wordlists,
            {k: v[0] for k, v in asjp.items() if k in grambank},
            converter,
            load=load,
            threshold=MIN_LANGS)

        gb_dic = dict(grambank.items())
        gb_wl = convert_data(
            gb_dic,
            {k: v[0] for k, v in asjp.items()},
            gb_conv,
            load='grambank',
            threshold=1)

        # Combine data vectors
        for lang in data:
            data[lang][2] = data[lang][2] + gb_wl[lang][2]

    # Add Carari
    if test_isolates and input == 'lexibank':
        data['cara1273'] = convert_data(
           dict(get_other(mode="carari").items()),
           {k: v[0] for k, v in asjp.items()},
           converter,
           threshold=1,
           load="lexical"
           )

    # Prepare split
    features = []
    labels = []
    idx = 0

    fam2w = defaultdict(int)
    idx2fam = dict(enumerate(set((data[lang][0] for lang in data))))
    fam2idx = {family: idx for idx, family in enumerate(set(data[lang][0] for lang in data))}

    f1_macro = F1Score(num_classes=len(idx2fam), average='macro', task="multiclass")
    f1_micro = F1Score(num_classes=len(idx2fam), average='micro', task="multiclass")

    for lang in data:
        family = data[lang][0]
        fam2w[family] = fam2w.get(family, 0) + 1
        # Add test cases to test list
        if (test_longdistance and family in ['Sino-Tibetan', 'Uto-Aztecan', 'Indo-European'] and
                any(lang in x for x in [sinitic, northern_uto, anatolian, tocharian])) or \
                test_isolates and lang in isolates:
            tests[lang] = data[lang]
        else:
            features.append(data[lang][2])
            labels.append(fam2idx[family])

    # Summary stats
    summary_stats = {
        'Number of families': len(fam2idx),
        'Size of vector': len(data[lang][2]),
        'Number of concepts': len(data[lang][2]) / n_pars
    }
    print(summary_stats)

    # Weights
    class_weights = [round(fam2w[max(fam2w, key=fam2w.get)] / fam2w[fam], 3) for fam in fam2w]
    class_weights = torch.FloatTensor(class_weights)
    class_weights = class_weights.to(device)

    # Data to tensor
    features = torch.Tensor(np.array(features))
    labels = torch.LongTensor(np.array(labels))
    features = features.to(device)
    labels = labels.to(device)
    tensor_ds = TensorDataset(features, labels)

    # Model hyperparameters
    input_dim = features.size()[1]  # Length of data tensor
    hidden_dim = HIDDEN*len(idx2fam)
    output_dim = len(idx2fam)

    class FF(nn.Module):
        """Network model with functions for forward-pass and predictions."""
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
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

        def predict(self, vector, language):
            """Predicts based on new data."""
            vector = torch.Tensor(np.array([vector[language][2]]))
            vector = vector.to(device)

            outs = self(vector)
            _, prediction = torch.max(outs.data, 1)
            prediction = idx2fam[prediction.item()]

            return prediction

    for run in range(runs):
        print('--- New Run: ', run, '/', runs, '---')
        fam_final = defaultdict()
        train_dataset, test_dataset = random_split(tensor_ds, [0.80, 0.20])
        # Alternative to weighted loss function: Weighted Sampler; however, this works less well
        # weights = []
        # for _, label in train_dataset:
        #     weights.append(class_weights[label])
        # weights = torch.Tensor(weights)
        # weights = weights.to(device)
        # sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH)

        model = FF(input_dim, hidden_dim, output_dim)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        no_improve = 0
        fam_high = 0
        best_macro = 0
        iters = 0

        for _ in range(EPOCHS):
            if no_improve < 20:
                for idx, (train_data, labels) in enumerate(train_loader):
                    optimizer.zero_grad()   # Clear gradients
                    outputs = model(train_data)   # Forward pass to get output/logits
                    loss = criterion(outputs, labels)
                    loss.backward()         # Getting gradients w.r.t. parameters
                    optimizer.step()        # Updating parameters

                    # Calculate Accuracy for test set
                    iters += 1
                    if iters % 50 == 0:
                        family_results = defaultdict()
                        fam_avg = defaultdict()
                        for test_data, labels in test_loader:
                            outputs = model(test_data)
                            _, predicted = torch.max(outputs.data, 1)

                            macro = round(f1_macro(predicted.cpu(), labels.cpu()).item(), 10)
                            micro = round(f1_micro(predicted.cpu(), labels.cpu()).item(), 10)

                            print('F1-Score macro: ', macro)
                            print('F1-Score micro: ', micro)

                            for idx, label in enumerate(labels):
                                pred = int(predicted[idx])
                                label = int(label)
                                # Labels per family
                                if label in family_results:
                                    family_results[label].append(pred)
                                else:
                                    family_results[label] = [pred]

                        for fam in family_results:
                            corr = sum([1 for pred in family_results[fam] if fam == pred])
                            total = len(family_results[fam])
                            fam_avg[idx2fam[fam]] = [100 * corr / total, total]

                        fam_acc = mean(fam_avg[k][0] for k in fam_avg)
                        print(f'Iter: {iters}. Loss: {round(loss.item(), 5)}. F1 Macro: {round(macro, 5)}.')
                        # if fam_acc > fam_high:
                        #     fam_high = fam_acc
                        #     fam_final = fam_avg
                        #     no_improve = 0
                        #     torch.save(model.state_dict(), 'best-mpar.pt')
                        if macro > best_macro:
                            best_macro = macro
                            fam_high = fam_acc
                            fam_final = fam_avg
                            no_improve = 0
                            torch.save(model.state_dict(), 'best-mpar.pt')
                        else:
                            no_improve += 1

        fam_scores.append(int(fam_high))
        f1_scores.append(100*best_macro)
        print(best_macro)
        print(f1_scores)

        model.load_state_dict(torch.load('best-mpar.pt', weights_only=True))

        # Compute cosine distances for families
        if distances is True:
            dist = [[0.0 for f in fam2idx] for f in fam2idx]
            weights = list(model.parameters())
            w = weights[4]  # Matrix with length fam2idx
            # print(w.shape)  # first dimension must be len(fam2idx)

            for i, v in enumerate(w):
                v = v.cpu().detach()
                for j, u in enumerate(w):
                    u = u.cpu().detach()
                    dist[i][j] = distance.cosine(v, u)

            with open('family-distances.tsv', 'w', encoding='utf8') as f:
                f.write(' '+str(len(fam2idx))+'\n')
                for i, row in enumerate(dist):
                    f.write(slug(idx2fam[i], lowercase=False) + ' ')
                    f.write(' '.join([f'{cell}' for cell in row])+'\n')

        # Add family results
        for fam in fam_final:
            results_per_fam.setdefault(fam, []).append([
                run,
                fam,
                fam2w[fam],
                fam_final[fam][1],
                round(fam_final[fam][0], 3)
            ])
        results_per_fam.setdefault(fam, []).append([
                run,
                'Total',
                len(data[lang][2]),
                len(labels),
                best_macro
        ])

        # Test experiments
        for lang in tests:
            results[lang, tests[lang][0]] = [model.predict(tests, lang)]

    print('---------------')
    results_table = []
    for item in results:
        results_str = '\n'.join(
            f"{k}: {v}"
            for k, v in Counter(results[item]).items()
            if v > (len(results[item])*0.05)
        )
        results_table.append([item[0], item[1], results_str])

        # Used for grid output
        # results_table.append(SEPARATING_LINE)

    print(tabulate(
        results_table,
        headers=[
            'Language',
            'Family',
            'Predictions'
            ],
        tablefmt='pipe'
        ))

    print('---------------')
    for fam, rows in sorted(results_per_fam.items()):
        table += [[
            fam,
            mean([r[2] for r in rows]),
            round(mean([r[3] for r in rows]), 1),   # Tested langs
            round(mean([r[4] for r in rows]), 2),   # Acc
            0 if len(rows) < 2 else round(stdev([r[4] for r in rows]), 2)  # SD of accuracy
            ]]

    table += [[
        'TOTAL',
        len(data),
        len(test_dataset),
        round(mean(fam_scores), 2),
        round(stdev(fam_scores), 2)
        ]]

    table += [[
        'TOTAL f1',
        len(data),
        len(test_dataset),
        round(mean(f1_scores), 2),
        round(stdev(f1_scores), 2)
        ]]

    header = ['Family', 'Languages', 'Tested', 'Avg. Fam. Accuracy', 'Fam-STD']
    # Detailed results per run
    output = 'results/results_' + input + '.tsv'
    with open(output, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Run', 'Family', 'Languages', 'Tested', 'Accuracy'])
        for family, run in results_per_fam.items():
            writer.writerows(run)

    print(tabulate(
        table,
        headers=header,
        floatfmt='.2f',
        tablefmt='pipe'
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=100,
                        help='The number of iterations the model should run. We recommend n>10')
    parser.add_argument('--input', type=str,
                        help='Choose the dataset: lexibank, grambank, or combined')
    parser.add_argument('-isolates', action='store_true')
    parser.add_argument('-longdistance', action='store_true')
    parser.add_argument('-distances', action='store_true',
                        help='Adds the cosine distances of the model weights for each family')

    args = parser.parse_args()

    run_ala(
        input=args.input,
        runs=args.runs,
        test_isolates=args.isolates,
        test_longdistance=args.longdistance,
        distances=args.distances
        )
