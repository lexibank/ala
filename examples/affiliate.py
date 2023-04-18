from ala import get_wordlists, affiliate_by_consonant_class, get_asjp, training_data, get_lingpy
import random
import statistics
from tabulate import tabulate
from tqdm import tqdm

THRESHOLD=0.03

min_classes = 5
asjp = get_asjp()
wordlists = get_wordlists("lexibank.sqlite3")
train, test = training_data(
        wordlists, 
        {k: v[0] for k, v in get_asjp().items()}, 
        0.8,
        min_classes
        )
hits, scores = [], []
by_fam_table = []
for fam, data in tqdm(test.items()):
    selected = random.sample(
            [gcode for gcode in data], 
            5 if 5 <= len(data) else len(data)
            )
    fam_hits, fam_scores = [], []
    for gcode, itms in [(a, b) for a, b in data.items() if a in selected]:
        wl = get_lingpy(
            itms, 
            ["lid", "doculect", "family", "concept", "tokens", "cog"])
        fams = affiliate_by_consonant_class(
                gcode, 
                wl, 
                train, 
                criterion="max"
                )
        best_fam = fams[0][0]
        if best_fam == "Unclassified":
            best_fam = ""
        best_fam_score = fams[0][1]
        if best_fam_score <= THRESHOLD:
            best_fam = "Unclassified"
        if fam == best_fam:
            hits += [1]
            scores += [best_fam_score]
            fam_hits += [1]
            fam_scores += [best_fam_score]
        else:
            hits += [0]
            fam_hits += [0]
            fam_scores += [best_fam_score]
            scores += [best_fam_score]
    by_fam_table += [[
        fam, len(selected), 
        statistics.mean(fam_scores),
        statistics.mean([s for h, s in zip(fam_hits, fam_scores) if h == 1] or [0]),
        statistics.mean([s for h, s in zip(fam_hits, fam_scores) if h == 0] or [0]),
        statistics.mean(fam_hits)]]

by_fam_table += [[
    "TOTAL", len(hits), statistics.mean(scores), 
    statistics.mean([s for h, s in zip(hits, scores) if h == 1]),
    statistics.mean([s for h, s in zip(hits, scores) if h == 0]),
    statistics.mean(hits)]]
print(tabulate(
    by_fam_table,
    headers=[
        "Family", "Samples", 
        "Scores",
        "Scores (Hits)",
        "Scores (Fails)",
        "Accuracy"],
    floatfmt=".2f",
    tablefmt="pipe"))
        
