# https://matplotlib.org/stable/gallery/statistics/violinplot.html#sphx-glr-gallery-statistics-violinplot-py
# https://stackoverflow.com/questions/29776114/half-violin-plot-in-matplotlib

from matplotlib import pyplot as plt
import csv
from collections import defaultdict
from pathlib import Path
import statistics


datasets = [
        "asjp-holm-max",
        "lb-swa-100-max",
        "asjp",
        "grambank",
        "lexibank",
        "combined"
        ]

methods = {
        "asjp-holm-max": "ASJP Baseline",
        "lb-star-max": "Lexibank Baseline (Starostin 110)",
        "lb-swa-100-max": "Lexibank Baseline (Swadesh 100)",
        "lb-swa-200-max": "Lexibank Baseline (Swadesh 200)",
        "lj-max": "Lexibank Baseline (Leipzig-Jakarta)",
        "asjp": "ASJP ALA Model",
        "combined": "Combined ALA Model",
        "grambank": "Grambank ALA Model",
        "lexibank": "Lexibank ALA Model",
        }


data = defaultdict(lambda: defaultdict(list))
for ds in datasets:
    with open(Path(__file__).parent / "results" / str("results_" + ds + ".tsv"),
              encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            data[ds][row[0]] += [float(row[-1])]

runs = {ds: [statistics.mean(vals) for vals in data[ds].values()] for ds in datasets}
for ds in datasets:
    print(ds, statistics.mean(runs[ds]))

plt.clf()
fig, axs = plt.subplots(1, 1, layout="constrained", figsize=(12, 4))

for i, ds in enumerate(datasets[::-1]):
    plt.violinplot(runs[ds], [i * 0.5 + 1], points=100, quantiles=[0.25, 0.75], showmeans=True,
                   side="high", orientation="horizontal")
plt.yticks([i * 0.5 + 1 for i in range(len(datasets))],
           [methods[d] for d in datasets[::-1]], fontsize=15)
plt.xlim(50, 100)
plt.savefig(Path(__file__).parent / "plots" / "violin-scores.pdf")

# Alternative plot with color legend
# plt.yticks([])
# handles = [Line2D([0], [0], marker='o', color='w', label=ds.upper(),
#                   markerfacecolor=f'C{i}', markersize=15)
#            for i, ds in enumerate(datasets[::-1])]

# plt.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)
# plt.savefig(Path(__file__).parent / "plots" / "violin-scores-alternative.pdf", dpi=300)
