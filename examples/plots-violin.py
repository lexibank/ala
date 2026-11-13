# https://matplotlib.org/stable/gallery/statistics/violinplot.html#sphx-glr-gallery-statistics-violinplot-py
# https://stackoverflow.com/questions/29776114/half-violin-plot-in-matplotlib

from matplotlib import pyplot as plt
import csv
from collections import defaultdict
from pathlib import Path
import statistics


datasets = [
    # Baseline comparison
    #"bl_asjp-holm-max",
    #"bl_lb-swa-100-max",
    #"asjp",
    #"grambank",
    #"lexibank",
    #"combined"

    # Conceptlist Comparison
    # "lexibank_starostin",
    # "lexibank_swadesh100",
    # "lexibank_swadesh200",
    # "lexibank_tadmor100"

    # Lexibank features
    "lexibank_cons_1",
    "lexibank_cons_2",
    "lexibank_feature_cons_1",
    "lexibank_feature_cons_2",
    "lexibank_feature_vowels_2",
    "lexibank_features_2",
    "lexibank_features_4"
    ]

methods = {
        "bl_asjp-holm-max": "ASJP Baseline",
        "bl_lb-star-max": "Lexibank Baseline (Starostin 110)",
        "bl_lb-swa-100-max": "Lexibank Baseline",
        "bl_lb-swa-200-max": "Lexibank Baseline (Swadesh 200)",
        "bl_lj-max": "Lexibank Baseline (Leipzig-Jakarta)",
        "asjp": "ASJP ALA Model",
        "combined": "Combined ALA Model",
        "grambank": "Grambank ALA Model",
        "lexibank": "Lexibank ALA Model",
        "lexibank_starostin": "Starostin-110",
        "lexibank_swadesh100": "Swadesh-100",
        "lexibank_swadesh200": "Swadesh-200",
        "lexibank_tadmor100": "Tadmor-100",
        "lexibank_cons_2": "Two Consonants (Dolgo)",
        "lexibank_cons_1": "One Consonant (Dolgo)",
        "lexibank_feature_cons_1": "One Consonant (Features)",
        "lexibank_feature_cons_2": "Two Consonants (Features)",
        "lexibank_features_2": "Two Segments (Features)",
        "lexibank_features_4": "Four Segments (Features)",
        "lexibank_feature_vowels_2": "Two Vowels (Features)"
        }

# Compute avg per run
run_averages = {}
for ds in datasets:
    run_averages[ds] = {}
    with open(Path(__file__).parent / "results" / str("results_" + ds + ".tsv"),
              encoding='utf8') as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run_id = row['Run']
            value = float(row['Score'])
            if run_id not in run_averages[ds]:
                run_averages[ds][run_id] = []

            run_averages[ds][run_id].append(value)
    
    # Compute mean for each run
    run_averages[ds] = {run_id: statistics.mean(vals) for run_id, vals in run_averages[ds].items()}

# print(run_averages)

filtered_data = {}
for ds in datasets:
    all_run_means = list(run_averages[ds].values())
    sorted_vals = sorted(all_run_means)
    n = len(sorted_vals)
    
    # Calculate the 89% interval
    lower_idx = int(n * 0.055)
    upper_idx = int(n * 0.945)
    
    # Ensure we have at least one value
    if upper_idx <= lower_idx:
        upper_idx = lower_idx + 1
    
    filtered_data[ds] = sorted_vals[lower_idx:upper_idx]

filtered_data = dict(sorted(filtered_data.items(), key=lambda item: statistics.mean(item[1])))

# for ds in datasets:
#     # print(filtered_data[ds])
#     print(ds, statistics.mean(filtered_data[ds]))
print(filtered_data)

plt.clf()
fig, axs = plt.subplots(1, 1, layout="constrained", figsize=(12, 4))

for i, ds in enumerate(filtered_data):
    plt.violinplot(filtered_data[ds], [i * 0.5 + 1], points=20, 
                   quantiles=[0.25, 0.75], showmeans=True, side="high", orientation="horizontal")

plt.yticks([i * 0.5 + 1 for i in range(len(datasets))],
           [methods[d] for d in filtered_data], fontsize=15)
plt.xlim(0, 100)
plt.savefig(Path(__file__).parent / "plots" / "violin-scores.pdf")
