from ala.data import get_lb, get_other, get_gb, get_asjp
from ala.baseline import affiliate_by_consonant_class, get_sound_class
import collections
import statistics
import random
from tqdm import tqdm
from torchmetrics import F1Score
from torch import tensor
from pyconcepticon import Concepticon
from sys import argv

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

if ds == "asjp":
    clist = "Holman-2008-40"

fnames = {
        "Swadesh-1955-100": "swa-100",
        "Swadesh-1952-200": "swa-200",
        "Tadmor-2009-100": "lj",
        "Holman-2008-40": "holm",
        "Starostin-1991-110": "star"
        }

con = Concepticon()
holm = {concept.concepticon_gloss for concept in
         con.conceptlists["Holman-2008-40"].concepts.values()}
swad = {concept.concepticon_gloss for concept in 
        con.conceptlists[clist].concepts.values()}

lb = get_lb()
asjp = get_other(mode="asjp")
gb = get_gb()
asjpc = get_asjp()

common_languages = [lng for lng in lb if lng in asjp and lng in gb]
lb = {k: v for k, v in lb.items() if k in common_languages}
asjp = {k: v for k, v in asjp.items() if k in common_languages}

by_fam_lb = collections.defaultdict(dict)
by_fam_asjp = collections.defaultdict(dict)

for key, wl in lb.items():
    if key in common_languages:
        fam = asjpc[key][0]
        
        # filter concepts
        wl_ = {k: v for k, v in wl.items() if v[2] in swad}
        by_fam_lb[fam][key] = wl_
        wl_ = {k: v for k, v in asjp[key].items() if v[2] in holm}
        by_fam_asjp[fam][key] = wl_

# get unclassified 
unclassified = {fam for fam in by_fam_lb if len(by_fam_lb[fam]) == 1}
by_fam_lb["Unclassified"] = {}
by_fam_asjp["Unclassified"] = {}
for fam in unclassified:
    by_fam_lb["Unclassified"][list(by_fam_lb[fam])[0]] = list(by_fam_lb[fam].values())[0]
    del by_fam_lb[fam]
    by_fam_asjp["Unclassified"][list(by_fam_asjp[fam])[0]] = list(by_fam_asjp[fam].values())[0]
    del by_fam_asjp[fam]

# make subselection by eligible language families
by_fam_lb = {k: v for k, v in by_fam_lb.items() if len(v) >= MINS}
by_fam_asjp = {k: v for k, v in by_fam_asjp.items() if len(v) >= MINS}

# language to family
l2fam = {}
for fam in by_fam_lb:
    for lng in by_fam_lb[fam]:
        l2fam[lng] = fam

# add sound classes
for key in common_languages:
    if key in l2fam:
        fam = l2fam[key]
        for k, row in by_fam_lb[fam][key].items():
            row += [get_sound_class(row[2], row[3])]
            row[1] = fam
        for k, row in by_fam_asjp[fam][key].items():
            row += [get_sound_class(row[2], row[3])]
            row[1] = fam


print("Investigating {0} languages from {1} families.".format(
    len(l2fam), len(by_fam_asjp)))


print("start investigation")
# select families that occur in lexibank


if ds == "lb":
    by_fam = by_fam_lb
elif ds == "asjp":
    by_fam = by_fam_asjp
    
base_path = "results/results_" + ds + "-" + fnames[clist]

f = open(base_path + "-mean.tsv", "w")
f.write("\t".join(["Run", "Family", "Languages", "Tested", "Score"]) + "\n")
f.close()

f = open(base_path + "-max.tsv", "w")
f.write("\t".join(["Run", "Family", "Languages", "Tested", "Score"]) + "\n")
f.close()

f = open(base_path + "-fs.tsv", "w")
f.write("\t".join(["Run", "Score"]) + "\n")
f.close()

fscore = F1Score(task="multiclass", num_classes=len(by_fam))
fam2idx = {fam: idx + 1 for idx, fam in enumerate(by_fam)}
fscores = []
for i in range(RUNS):
    gs, ts = [], []
    results_mean, results_max = {}, {}
    selection_train, selection_test = {}, {}
    scores = {}
    for fam in by_fam:
        faml = len(by_fam[fam])
        if faml >= MINS:
            results_mean[fam] = []
            results_max[fam] = []
            trainl = int(faml * 0.8 + 0.5)
            languages = list(by_fam[fam])
            train = random.sample(languages, trainl)
            test = [l for l in languages if l not in train]
            for t in train:
                selection_train[t] = asjp[t]
            for t in test:
                selection_test[t] = asjp[t]
            scores[fam] = (len(train), len(test))
    for lng, wl in tqdm(selection_test.items()):
        res = affiliate_by_consonant_class(
                lng, wl, selection_train)
        if res[0][0] == l2fam[lng]:
            results_mean[l2fam[lng]] += [1]
        else:
            results_mean[l2fam[lng]] += [0]
        if res[2][0] == l2fam[lng]:
            results_max[l2fam[lng]] += [1]
        else:
            results_max[l2fam[lng]] += [0]
        gs += [fam2idx[l2fam[lng]]]
        ts += [fam2idx[res[2][0]]]
        
    
    f = open(base_path + "-mean.tsv", "a")
    total_1, total_2, total_3 = [], [], []
    for k, v in results_mean.items():
        f.write("\t".join([
            str(i + 1),
            str(k),
            str(sum(scores[k])),
            str(scores[k][1]),
            "{0:.2f}".format(100 * statistics.mean(v))]) + "\n")
        total_1 += [statistics.mean(v)]
    f.close()
    f = open(base_path + "-max.tsv", "a")
    for k, v in results_max.items():
        f.write("\t".join([
            str(i + 1),
            str(k),
            str(sum(scores[k])),
            str(scores[k][1]),
            "{0:.2f}".format(100 * statistics.mean(v))]) + "\n")
        total_2 += [statistics.mean(v)]

    f.close()
    f = open(base_path + "-fs.tsv", "a")
    fs = fscore(tensor(gs), tensor(ts))
    f.write("\t".join([
        str(i + 1),
        "{0:.2f}".format(100 * float(fs))]) + "\n")
    fscores += [float(fs)]
    f.close()
    print("Run {0}, mean: {1:.2f}, max: {2:.2f}, fs: {3:.2f}".format(
        i + 1,
        statistics.mean(total_1),
        statistics.mean(total_2),
        float(fs)))
    
f = open(base_path + "-fs.tsv", "a")
f.write("\t".join([
    "0",
    "{0:.2f}".format(100 * statistics.mean(fscores))]) + "\n")
f.close()


        
