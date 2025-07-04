from matplotlib import pyplot as plt
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path
from matplotlib import colormaps
import numpy as np


isolates = OrderedDict({
    "bang1363": "Bangime",
    "basq1248": "Basque",
    "kusu1250": "Kusunda",
    "mapu1245": "Mapudungun",
        })

datasets = [
        "lexibank",
        "grambank",
        "combined"
        ]


data = defaultdict(list)
for ds in datasets:
    with open(Path(__file__).parent / "results" / str("experiments_" + ds + ".tsv"),
              encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[0] in isolates:
                data[ds, isolates[row[0]]] += [(row[2], int(row[3]))]

# get top three for each data set
pies = {ds: {lng: {"labels": [], "hits": []} for lng in isolates.values()} for ds in datasets}
all_fams = {f: set() for f in datasets}
fams = set()
for (ds, lng), results in data.items():
    results = sorted(results, key=lambda x: x[1], reverse=True)
    top3 = results[:3]
    rest = sum([v[1] for v in results[3:]])
    for f, hits in top3:
        all_fams[ds].add(f)
        fams.add(f)
        pies[ds][lng]["labels"] += [f]
        pies[ds][lng]["hits"] += [hits]
    pies[ds][lng]["labels"] += ["Rest"]
    pies[ds][lng]["hits"] += [rest]

plt.clf()
cmap = colormaps.get_cmap("tab20")
cm = {}
fams = [f for f in sorted(fams) if f != "Unclassified"]
scale = np.linspace(0, 1, len(fams))
languages = list(isolates.values())

for i, _ in enumerate(fams):
    cm[fams[i]] = cmap(scale[i])

# black and white for major values
cm["Rest"] = "0.2"
cm["Unclassified"] = "0.9"

fig, axs = plt.subplots(len(datasets), len(isolates)+1, figsize=(10, 4),
                        # Sets the space between the individual plots
                        gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

for i, ds in enumerate(datasets):
    axs[i, 0].set_ylabel(ds.upper())
    axs[i, 4].axis("off")

    for j, lng in enumerate(languages):
        colors = [cm[f] for f in pies[ds][lng]["labels"]]
        axs[i, j].pie(pies[ds][lng]["hits"], colors=colors)
        axs[i, j].set_xlim(-1, 1)
        axs[i, j].set_ylim(-1, 1)
        if ds == "combined":
            axs[i, j].set_xlabel(lng)

for f, c in cm.items():
    axs[2, 4].plot(0, 0, "o", color=c, label=f)

axs[2, 4].legend(loc=3, fontsize=9)

plt.savefig(Path(__file__).parent / "plots" / "isolate-plot.pdf")
