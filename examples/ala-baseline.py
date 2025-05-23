from sys import argv
import collections
import statistics
import random
from tqdm import tqdm
from torch import tensor
from pyconcepticon import Concepticon
from ala.data import get_lb, get_gb, get_asjp, load_data
from ala.baseline import affiliate_by_consonant_class, get_sound_class

MINS = 5

try:
    clist = argv[3]
except:
    clist = "Swadesh-1955-100"

try:
    ds = argv[2]
except:
    ds = "lb"

try:
    RUNS = int(argv[1])
except:
    RUNS = 100

fnames = {
        "Swadesh-1955-100": "swa-100",
        "Swadesh-1952-200": "swa-200",
        "Tadmor-2009-100": "lj",
        "Holman-2008-40": "holm",
        "Starostin-1991-110": "star"
        }

con = Concepticon()
conc = {concept.concepticon_gloss for concept in con.conceptlists[clist].concepts.values()}

lb = get_lb()
gb = get_gb()

# Load ASJP family data
asjp_fams = get_asjp()
fams = {k: v[0] for k, v in asjp_fams.items()}

# Load ASJP wordlist data
asjp = load_data('asjp', 1)

# Filter for common languages
common_languages = [lng for lng in lb if lng in asjp_fams and lng in gb]

by_fam = collections.defaultdict(dict)
data = lb if ds == "lb" else asjp if ds == "asjp" else []
data = {k: v for k, v in data.items() if k in common_languages}

filtered_data = {}
# Filter concepts
for key, forms in data.items():
    if key in common_languages:
        fam = fams[key]
        wl_ = {k: v for k, v in forms.items() if v[2] in conc}
        by_fam[fam][key] = wl_
        data[key] = wl_

# get unclassified
unclassified = {fam for fam in by_fam if len(by_fam[fam]) == 1}

for fam in unclassified:
    by_fam["Unclassified"][list(by_fam[fam])[0]] = list(by_fam[fam].values())[0]
    del by_fam[fam]

# make subselection by eligible language families
by_fam = {k: v for k, v in by_fam.items() if len(v) >= MINS}
fam2idx = {fam: idx + 1 for idx, fam in enumerate(by_fam)}

# language to family
l2fam = {lng: fam for fam in by_fam for lng in by_fam[fam]}

# add sound classes
for key in common_languages:
    if key in l2fam:
        fam = l2fam[key]
        for k, row in by_fam[fam][key].items():
            row += [get_sound_class(row[2], row[3])]
            row[1] = fam

print(f"Investigating {len(l2fam)} languages from {len(by_fam)} families.")
print("start investigation")

base_path = "results/results_bl_" + ds + "-" + fnames[clist]
for i in range(RUNS):
    results_mean, results_max = {}, {}
    selection_train, selection_test = {}, {}
    total_1, total_2 = [], []
    scores = {}
    for fam in by_fam:
        results_mean[fam] = []
        results_max[fam] = []

        languages = list(by_fam[fam])
        sample_n = int(len(by_fam[fam]) * 0.8)

        train = random.sample(languages, sample_n)
        test = [l for l in languages if l not in train]

        for t in train:
            selection_train[t] = data[t]
        for t in test:
            selection_test[t] = data[t]

        scores[fam] = (len(train), len(test))

    for lng, wl in tqdm(selection_test.items()):
        res = affiliate_by_consonant_class(lng, wl, selection_train)

        results_mean[l2fam[lng]] += [1] if res[0][0] == l2fam[lng] else [0]
        results_max[l2fam[lng]] += [1] if res[2][0] == l2fam[lng] else [0]

    # Write mean results
    with open(base_path + "-mean.tsv", "a", encoding='utf8') as f:
        for k, v in results_mean.items():
            f.write("\t".join([
                str(i + 1),
                str(k),
                str(sum(scores[k])),
                str(scores[k][1]),
                f"{round(100 * statistics.mean(v), 2)}"]) + "\n")
            total_1 += [statistics.mean(v)]

    # Write max results
    with open(base_path + "-max.tsv", "a", encoding='utf8') as f:
        for k, v in results_max.items():
            f.write("\t".join([
                str(i + 1),
                str(k),
                str(sum(scores[k])),
                str(scores[k][1]),
                f"{round(100 * statistics.mean(v), 3)}"]) + "\n")
            total_2 += [statistics.mean(v)]

    # Print output per run
    results = {
        "Mean": round(statistics.mean(total_1), 3),
        "Max": round(statistics.mean(total_2), 3)
    }
    print(f"Run {i + 1}: {results}")
