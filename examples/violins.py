# https://matplotlib.org/stable/gallery/statistics/violinplot.html#sphx-glr-gallery-statistics-violinplot-py

# https://stackoverflow.com/questions/29776114/half-violin-plot-in-matplotlib

from matplotlib import pyplot as plt
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path
from matplotlib.pyplot import subplot
from matplotlib import colormaps
import numpy as np
import statistics


datasets = [
        #"asjp-holm-mean",
        "asjp-holm-max",
        #"lb-swa-100-mean",
        "lb-swa-100-max",
        "lb-star-max",
        #"lb-swa-200-mean",
        "lb-swa-200-max",
        #"lb-lj-100-mean",
        "lb-lj-max",
        #"asjp",
        #"lexibank",
        #"lexibank-200",
        #"grambank",
        #"combined"
        ]


data = defaultdict(lambda : defaultdict(list))
for ds in datasets:
    with open(Path(__file__).parent / "results" / str("results_" + ds
                                                      +".tsv")) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[0] != "TOTAL":
                data[ds][row[0]] += [float(row[-1])]
runs = {ds: [] for ds in datasets}
for ds in datasets:
    for k, vals in data[ds].items():
        runs[ds] += [statistics.mean(vals)]

plt.clf()
fig, axs = plt.subplots(1, 1, layout="constrained", figsize=(12, 4))

for i, ds in enumerate(datasets[::-1]):
    plt.violinplot(runs[ds], [i * 0.5 + 1], points=100, quantiles=[0.25, 0.75], showmeans=True, side="high", orientation="horizontal")
plt.yticks([i * 0.5 + 1 for i in range(len(datasets))], [d.upper() for d in
                                             datasets[::-1]], fontsize=15)
plt.xlim(50, 100)
plt.savefig(Path(__file__).parent / "plots" / "violin-scores.pdf")


