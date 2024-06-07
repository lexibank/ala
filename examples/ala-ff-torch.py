import argparse
import csv
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


def run_ala(data, intersection=False, test_isolates=False, test_np=False,
            test_longdistance=False, distances=False):
    """Defines the workflow for data loading in the different settings."""
    # Hyperparameters
    runs = 100
    epochs = 5000
    batch = 2096
    hidden = 4  # multiplier for length of fam
    learning_rate = 1e-3
    min_langs = 5

    tests = defaultdict()
    if test_longdistance is True:
        northern_uto = extract_branch(gcode='nort2953')
        anatolian = extract_branch(gcode='anat1257')
        tocharian = extract_branch(gcode='tokh1241')
        sinitic = extract_branch(gcode='sini1245')

    isolates = ['bang1363', 'basq1248', 'mapu1245', 'kusu1250'] if test_isolates is True else []

    # Switch on GPU if available
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Current device:', device)

    # Empty lists and dicts for results
    table = []
    fam_scores = []

    results_per_fam = defaultdict()  # store family results
    results = defaultdict()  # store experiment results

    # Setup for databases
    asjp = get_asjp()
    grambank = get_gb()
    lexibank = get_lb()
    gb_conv = feature2vec(get_db('data/grambank.sqlite3'))
    lb_conv = concept2vec(get_db('data/lexibank.sqlite3'), model='dolgo')

    load = 'grambank' if data == 'grambank' else 'lexical'
    converter = gb_conv if data == 'grambank' else lb_conv

    if data in ('lexibank', 'lb_mod', 'combined'):
        wordlists = dict(lexibank.items())
    elif data == 'grambank':
        wordlists = dict(grambank.items())
    elif data == 'asjp':
        wordlists = dict(get_other(mode='asjp').items())

    if intersection is True:
        intersec = lexibank if data == 'grambank' else grambank
        wordlists = {k: wordlists[k] for k in wordlists if k in intersec}
        mod = '_intersec'
    else:
        mod = '_no'

    if data != 'combined':
        full_data = convert_data(
            wordlists,
            {k: v[0] for k, v in asjp.items()},
            converter,
            load=load,
            threshold=min_langs)

    # test integration of ASJP Genus
    # test = {k: v[1] for k, v in get_asjp().items()}
    # print(test)

    elif data == 'combined':
        full_data = convert_data(
            wordlists,
            {k: v[0] for k, v in asjp.items() if k in grambank},
            converter,
            load=load,
            threshold=min_langs)

        gb_dic = dict(grambank.items())
        gb_wl = convert_data(
            gb_dic,
            {k: v[0] for k, v in asjp.items()},
            gb_conv,
            load='grambank',
            threshold=1)
        # Combine data vectors
        for lang in full_data:
            full_data[lang][2] = full_data[lang][2] + gb_wl[lang][2]

    if test_np is True:
        train_np = ['zapa1253', 'iqui1243', 'ando1255', 'arab1268', 'agua1253', 'achu1248', 'shua1257']
        np_wl = dict(get_other(mode='np').items())

        np_data = convert_data(np_wl, {k: v[0] for k, v in asjp.items()}, converter, load='np')
        for lang in np_data:
            if lang in train_np:
                full_data[lang] = np_data[lang]
            else:
                tests[lang] = np_data[lang]

    if test_longdistance is True:
        iecor_wl = dict(get_other(mode='iecor').items())

        if intersection is True:
            iecor_wl = {k: iecor_wl[k] for k in iecor_wl if k in intersec}

    features = []
    labels = []
    idx2fam = defaultdict()
    fam2idx = defaultdict()
    fam2weight = defaultdict()
    idx = 0

    for lang in full_data:
        family = full_data[lang][0]
        if family not in fam2idx:
            idx2fam[idx] = family
            fam2idx[family] = idx
            fam2weight[family] = 1
            idx += 1
        else:
            fam2weight[family] += 1

        if test_longdistance is True and family in ['Sino-Tibetan', 'Uto-Aztecan', 'Indo-European']:
            if lang in sinitic or lang in northern_uto or lang in anatolian or lang in tocharian:
                tests[lang] = full_data[lang]
            else:
                features.append(full_data[lang][2])
                labels.append(fam2idx[family])

        # Add test cases to test and others out
        elif family == 'Unclassified' and test_isolates is True:
            if lang in isolates:
                tests[lang] = full_data[lang]
            else:
                features.append(full_data[lang][2])
                labels.append(fam2idx[family])

        else:
            features.append(full_data[lang][2])
            labels.append(fam2idx[family])

    # Weights
    largest_class = fam2weight[max(fam2weight, key=fam2weight.get)]
    class_weights = [round(largest_class / fam2weight[fam], 3) for fam in fam2weight]
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
    hidden_dim = hidden*len(idx2fam)
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

            outs = model(vector)
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

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch,
            shuffle=True
            )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch
            )

        model = FF(input_dim, hidden_dim, output_dim)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        iters = 0
        fam_high = 0
        no_improve = 0

        for _ in range(epochs):
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
                        print(f'Iter: {iters}. Loss: {round(loss.item(), 9)}. Average family accuracy: {round(fam_acc, 3)}')
                        if fam_acc > fam_high:
                            fam_high = fam_acc
                            fam_final = fam_avg
                            no_improve = 0
                            torch.save(model.state_dict(), 'best-model-parameters.pt')
                        else:
                            no_improve += 1

        fam_scores.append(int(fam_high))
        model.load_state_dict(torch.load('best-model-parameters.pt'))

        # Compute cosine distances for families
        if distances is True:
            dist = [[0.0 for f in fam2idx] for f in fam2idx]
            weights = list(model.parameters())
            w = weights[4]  # Matrix with length fam2idx
            # print(w.shape)  # first dimension must be len(fam2idx)

            for i, v in enumerate(w):
                v = v.cpu().detach().numpy()
                for j, u in enumerate(w):
                    u = u.cpu().detach().numpy()
                    dist[i][j] = distance.cosine(v, u)

            with open('family-distances.tsv', 'w', encoding='utf8') as f:
                f.write(' '+str(len(fam2idx))+'\n')
                for i, row in enumerate(dist):
                    f.write(slug(idx2fam[i], lowercase=False) + ' ')
                    f.write(' '.join([f'{cell}' for cell in row])+'\n')

        # Add family results
        for fam in fam_final:
            if fam in results_per_fam:
                results_per_fam[fam].append([
                    run,                            # Number of run
                    fam,                            # Language Family
                    fam2weight[fam],                # Number of langs in fam
                    fam_final[fam][1],              # Number of langs tested
                    round(fam_final[fam][0], 3)     # Accuracy
                ])

            else:
                results_per_fam[fam] = [[
                    run,                            # Number of run
                    fam,                            # Language Family
                    fam2weight[fam],                # Number of langs in fam
                    fam_final[fam][1],              # Number of langs tested
                    round(fam_final[fam][0], 3)     # Accuracy
                    ]]

        # Test experiments
        for lang in tests:
            results_id = (lang, tests[lang][0])
            pred = model.predict(tests, lang)
            if results_id in results:
                results[results_id].append(pred)
            else:
                results[results_id] = [pred]

    print('---------------')
    results_table = []
    for item in results:
        results_str = ''
        for i, (k, v) in enumerate(Counter(results[item]).items()):
            sep = '' if i < 0 else '\n'
            if v > (len(results[item])*0.05):
                results_str = results_str + k + ': ' + str(v) + sep

        results_table.append([
            item[0],
            item[1],
            results_str
            ])
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
            round(stdev([r[4] for r in rows]), 2),  # SD of accuracy
            ]]

    table += [[
        'TOTAL',
        len(full_data),
        len(test_dataset),
        round(mean(fam_scores), 2),
        round(stdev(fam_scores), 2)
        ]]

    header = ['Family', 'Languages', 'Tested', 'Avg. Fam. Accuracy', 'Fam-STD']
    # output = 'results/results_' + data + mod + '.tsv'
    # with open(output, 'w', encoding='utf8', newline='') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerow(header)
    #     writer.writerows(table)

    # # Detailed results per run
    # output_detailed = output.replace('.tsv', '_detailed.tsv')
    # with open(output_detailed, 'w', encoding='utf8', newline='') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerow(['Run', 'Family', 'Languages', 'Tested', 'Accuracy'])
    #     for family in results_per_fam:
    #         for run in results_per_fam[family]:
    #             writer.writerow(run)

    print(tabulate(
        table,
        headers=header,
        floatfmt='.2f',
        tablefmt='pipe'
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        help='Choose the dataset for your experiment: \
                            lexibank, grambank, or combined')
    parser.add_argument('-intersection', action='store_true',
                        help='Choose if intersect with another dataset')
    parser.add_argument('-isolates', action='store_true')
    parser.add_argument('-longdistance', action='store_true')
    parser.add_argument('-test_np', action='store_true')
    parser.add_argument('-distances', action='store_true',
                        help='Adds the cosine distances of the model weights for each family')

    args = parser.parse_args()

    run_ala(
        data=args.data,
        intersection=args.intersection,
        test_isolates=args.isolates,
        test_longdistance=args.longdistance,
        distances=args.distances,
        test_np=args.test_np
        )
