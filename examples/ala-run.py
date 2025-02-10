"""
Module for running the experiments for Automated Language Affiliation.
"""
import argparse
import csv
import sys
from collections import defaultdict, Counter
from statistics import mean, stdev
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import F1Score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from clldutils.clilib import add_format, Table
from ala import get_lb, get_asjp, get_gb, convert_data, get_other
from ala import concept2vec, feature2vec, get_db, extract_branch
from pyconcepticon import Concepticon

# Hyperparameters
EPOCHS = 5000
BATCH = 2048
HIDDEN = 4  # multiplier for length of fam
LEARNING_RATE = 1e-3



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

    def predict(self, vector, language):
        """Predicts based on new data."""
        vector = torch.Tensor(np.array([vector[language][2]]))
        vector = vector.to(device)

        outs = self(vector)
        _, prediction = torch.max(outs.data, 1)
        prediction = idx2fam[prediction.item()]

        return prediction



def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # Switch on GPU if available
    device = 'mps' if torch.backends.mps.is_available() \
            else 'cuda' if torch.cuda.is_available() \
            else 'cpu'
    print('Current device:', device)

    result_per_fam = defaultdict(list)
    results = defaultdict(list)

    # Setup for databases
    asjp = get_asjp()
    gb = get_gb()
    lb = get_lb()
    asjpb = get_other(mode="asjp")

    # get configuration options
    gb_conv, gb_keys = feature2vec(get_db('data/grambank.sqlite3'))
    lb_conv, lb_keys = concept2vec(
            get_db('data/lexibank.sqlite3'), model='dolgo',
            conceptlist="Swadesh-1955-100")
    asjp_conv, asjp_keys = concept2vec(
            get_db('data/lexibank.sqlite3'), model='dolgo',
            conceptlist="Holman-2008-40")

    # prepare datasets, only use common languages
    common_languages = [lng for lng in asjpb if lng in gb and lng in lb]
    print("There are {0} common languages in all three datasets.".format(len(common_languages)))

    # make selection of models for each datasets
    asjpb = {k: v for k, v in asjpb.items() if k in common_languages}
    gb = {k: v for k, v in gb.items() if k in common_languages}
    lb = {k: v for k, v in lb.items() if k in common_languages}

    # get only those words from lexibank that conform to Swadesh 100
    concepticon = Concepticon() # TODO add concepticon key
    swad100 = {concept.concepticon_gloss for concept in
               concepticon.conceptlists["Swadesh-1955-100"].concepts.values()}
    asjp40 = {concept.concepticon_gloss for concept in
              concepticon.conceptlists["Holman-2008-40"].concepts.values()}
    for k, wl in lb.items():
        lb[k] = {idx: row for idx, row in wl.items() if row[2] in swad100}
    for k, wl in asjpb.items():
        asjpb[k] = {idx: row for idx, row in wl.items() if row[2] in asjp40}

    print("Datasets have been reduced to the common languages.")

    # get number of families to be inferred
    language_families = defaultdict(int)
    for k, wl in lb.items():
        language_families[list(wl.values())[0][1]] += 1
    selected_families = {fam for fam, num in language_families.items() if num
                         >= args.minimum}
    selected_languages = []
    for k, wl in lb.items():
        fam = list(wl.values())[0][1]
        if fam in selected_families:
            selected_languages += [k]

    print("There are {0} languages from {1} families that can be tested.".format(
        len(selected_languages),
        len(selected_families)))

    
    load = 'grambank' if args.database == 'grambank' else 'lexical'
    conv = gb_conv if args.database == 'grambank' else lb_conv
    if args.database == "asjp":
        conv = asjp_conv

    # Load main data
    data_map = {
        'lexibank': lb,
        #'lb_mod': lb,
        'combined': lb,
        'grambank': gb,
        'asjp': asjpb
    }

    if args.database not in data_map:
        print("Invalid data selection. Please choose 'lexibank', 'grambank', 'combined' or 'asjp'")
        sys.exit()

    wl = data_map[args.database]
    n_pars = gb_keys if args.database == 'grambank' else gb_keys + lb_keys if args.database == 'combined' else lb_keys

    if args.database != 'combined':
        data = convert_data(wl, {k: v[0] for k, v in asjp.items()},
                            conv, load=load, threshold=args.minimum)

    # Set up combination of LB and GB
    # TODO: can throw errors if GB lacks a language, must take true subset
    elif args.database == 'combined':
        data = convert_data(wl, {k: v[0] for k, v in asjp.items() if k in gb},
                            conv, load=load, threshold=args.minimum)
        gb_wl = convert_data(data_map['grambank'], {k: v[0] for k, v in asjp.items()},
                             gb_conv, load='grambank', threshold=args.minimum)

        # Combine data vectors
        for lang in data:
            data[lang][2] += gb_wl[lang][2]

    idx2fam = dict(enumerate(set((data[lang][0] for lang in data))))
    fam2idx = {family: idx for idx, family in enumerate(set(data[lang][0] for lang in data))}
    # this computes F1 scores but only for part of the data, otherwise, we have
    # the balanced accuracy in the output in text files
    f1_macro = F1Score(num_classes=len(idx2fam), average='macro', task="multiclass")

    #test_langs = {lang: data[lang] for lang in data if (args.experiment and (
    #    any(lang in x for x in [sinitic, northern_uto, anatolian, tocharian])
    #    or lang in isolates))}

    #tests = {lang: data[lang] for lang in data if lang in test_langs}
    test_langs = [] # delete XXX
    features = [data[lang][2] for lang in data if lang not in test_langs]
    labels = [fam2idx[data[lang][0]] for lang in data if lang not in test_langs]

    # Summary stats
    summary_stats = {
        'Number of families': len(fam2idx),
        'Number of languages': len(data),
        'Size of vector': len(features[0]),
        'Number of concepts': len(features[0]) / n_pars
    }
    print(summary_stats)

    # Weights for CrossEntropy
    fam2w = defaultdict(int)
    for family in [data[lang][0] for lang in data if lang not in test_langs]:
        fam2w[family] += 1

    class_weights = [round(fam2w[max(fam2w, key=fam2w.get)] / fam2w[fam], 3) for fam in fam2w]
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Data to tensor
    features = torch.Tensor(np.array(features))
    all_labels = torch.LongTensor(np.array(labels))
    tensor_ds = TensorDataset(features, all_labels)
    
    for run in range(args.runs):
        print('--- New Run: ', run+1, '/', args.runs, '---')
        fam_final = defaultdict()
        train_ds, test_ds = train_test_split(
            tensor_ds, test_size=0.2, stratify=all_labels
            )

        train_ds = [(item[0].to(device), item[1].to(device)) for item in train_ds]
        test_ds = [(item[0].to(device), item[1].to(device)) for item in test_ds]

        train_loader = DataLoader(dataset=train_ds, batch_size=BATCH, shuffle=True)
        test_loader = DataLoader(dataset=test_ds, batch_size=BATCH)

        model = FF(features.size()[1], HIDDEN*len(idx2fam), len(idx2fam)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        no_improve = 0
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

                    iters += 1
                    if iters % 100 == 0:
                        family_results = defaultdict()
                        fam_avg = defaultdict()
                        for test_data, labels in test_loader:
                            outputs = model(test_data)
                            _, preds = torch.max(outputs.data, 1)
                            macro = round(f1_macro(preds.cpu(), labels.cpu()).item(), 10)

                            # Labels per family
                            for idx, label in enumerate(labels):
                                family_results.setdefault(int(label), []).append(int(preds[idx]))

                        for fam in family_results:
                            corr = sum(1 for pred in family_results[fam] if fam == pred)
                            total = len(family_results[fam])
                            fam_avg[idx2fam[fam]] = [100 * corr / total, total]

                        print(f'Iter: {iters}. Loss: {round(loss.item(), 5)}. F1: {macro}.')

                        if macro > best_macro:
                            best_macro = macro
                            fam_final = fam_avg
                            no_improve = 0
                            torch.save(model.state_dict(), 'best-mpar.pt')
                        else:
                            no_improve += 1

        model.load_state_dict(torch.load('best-mpar.pt', weights_only=True))

        # Add family results
        for fam, res in fam_final.items():
            result_per_fam[fam].append(
                [run + 1, fam, fam2w[fam], res[1], round(res[0], 3)]
                )

        result_per_fam['TOTAL'].append(
                [run + 1, 'TOTAL', len(data), len(test_ds), 100 * best_macro])

    output = 'results/results_' + args.database + '.tsv'
    with open(output, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Run', 'Family', 'Languages', 'Tested', 'Score'])
        for fam, rows in result_per_fam.items():
            writer.writerows(rows)

    # Summary table for command line
    table = [[
        fam, mean([r[2] for r in rows]),
        round(mean([r[3] for r in rows]), 1),   # Tested langs
        round(mean([r[4] for r in rows]), 2),   # Acc
        0 if len(rows) < 2 else round(stdev([r[4] for r in rows]), 2)]
        for fam, rows in sorted(result_per_fam.items())
    ]

    total = [fam for fam in table if fam[0] == 'TOTAL']
    new_table = [fam for fam in table if fam[0] != 'TOTAL'] + total

    with Table(args, *['Family', 'Languages', 'Tested', 'Avg. Fam. Accuracy', 'Fam-STD']) as t:
        t.extend(new_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=100,
                        help='The number of iterations the model should run. We recommend n>10')
    parser.add_argument('--database', type=str,
                        help='Choose the dataset: lexibank, grambank, or combined')
    parser.add_argument('--experiment', action='store_true')
    parser.add_argument('--distances', action='store_true',
                        help='Adds the cosine distances of the model weights for each family')
    parser.add_argument("--minimum", action="store", type=int, default=5,
                        help="select the minimum number of languages per family")
    add_format(parser, default='simple')
    args = parser.parse_args()

    run_ala(args)
