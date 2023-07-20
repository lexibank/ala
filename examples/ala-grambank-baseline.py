from ala import get_wordlists, get_gb, affiliate_by_grambank, affiliate_by_consonant_class, get_asjp, training_data, get_lingpy
import random
import statistics
from tabulate import tabulate
from tqdm import tqdm

THRESHOLD = 0.03
LEVEL = 0
min_classes = 3
test_size = 10
RUNS = 100
tt_split = 0.8
TEST_SIZE = 250

asjp = get_asjp()
wordlists = get_gb("grambank.sqlite3")
control = get_wordlists("lexibank.sqlite3")
wordlists = {k: v for k, v in wordlists.items() if k in control}

results = {"TOTAL": []}

for i in range(RUNS):

    train, test = training_data(
            wordlists, 
            {k: v[LEVEL] for k, v in get_asjp().items()}, 
            tt_split,
            min_classes
            )
    
    hits, scores, by_fam_table = [], [], []
    train_count, test_count = 0, 0
    for fam, data in tqdm(test.items()):
        if fam not in results:
            results[fam] = []
        selected = random.sample(
                [gcode for gcode in data], 
                test_size if test_size <= len(data) else len(data)
                )
        fam_hits, fam_scores = [], []
        test_count += len(data)
        train_count += len(train[fam])
        for gcode, itms in [(a, b) for a, b in data.items() if a in selected]:
            selected = {k: itms[k] for k in random.sample(
                sorted(itms), TEST_SIZE if TEST_SIZE <= len(itms) else len(itms))}
            wl = get_lingpy(
                selected, 
                ["lid", "doculect", "concept", "tokens", "cog"])
            fams = affiliate_by_grambank(
                    gcode, 
                    wl, 
                    train, 
                    criterion="max"
                    )
            best_fam = fams[0][0]
            #if best_fam == "Unclassified":
            #    best_fam = ""
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
        results[fam] += [[
            len(train[fam]), len(data), len(selected), 
            statistics.mean(fam_scores),
            statistics.mean([s for h, s in zip(fam_hits, fam_scores) if h == 1] or [0]),
            statistics.mean([s for h, s in zip(fam_hits, fam_scores) if h == 0] or [0]),
            statistics.mean(fam_hits)]]
    results["TOTAL"] += [[    
        train_count, test_count, len(hits), statistics.mean(scores), 
        statistics.mean([s for h, s in zip(hits, scores) if h == 1]),
        statistics.mean([s for h, s in zip(hits, scores) if h == 0]),
        statistics.mean(hits)]]

table = []
for family, rows in sorted(results.items()):
    if family != "TOTAL":
        row = [
                family,
                statistics.mean([r[0] for r in rows]),
                statistics.mean([r[1] for r in rows]),
                statistics.mean([r[2] for r in rows]),
                statistics.mean([r[3] for r in rows]),
                statistics.stdev([r[3] for r in rows]),
                statistics.mean([r[4] for r in rows]),
                statistics.stdev([r[4] for r in rows]),
                statistics.mean([r[5] for r in rows]),
                statistics.stdev([r[5] for r in rows]),
                statistics.mean([r[6] for r in rows]),
                statistics.stdev([r[6] for r in rows])
                ]
        table += [row]
table += [[
    "TOTAL",
    statistics.mean([row[0] for row in results["TOTAL"]]),
    statistics.mean([row[1] for row in results["TOTAL"]]),
    statistics.mean([row[2] for row in results["TOTAL"]]),
    statistics.mean([row[3] for row in results["TOTAL"]]),
    statistics.stdev([row[3] for row in results["TOTAL"]]),
    statistics.mean([row[4] for row in results["TOTAL"]]),
    statistics.stdev([row[4] for row in results["TOTAL"]]),
    statistics.mean([row[5] for row in results["TOTAL"]]),
    statistics.stdev([row[5] for row in results["TOTAL"]]),
    statistics.mean([row[6] for row in results["TOTAL"]]),
    statistics.stdev([row[6] for row in results["TOTAL"]])
    ]]


print(tabulate(
    table,
    headers=[
        "Family", 
        "Training",
        "Testing",
        "Samples",
        "Scores",
        "Sco-STD",
        "Scores (Hits)",
        "ScH-STD",
        "Scores (Fails)",
        "ScF-STD",
        "Accuracy",
        "Acc-STD"],
    floatfmt=".2f",
    tablefmt="pipe"))

THRESHOLD = 0.03
LEVEL = 0
min_classes = 3
test_size = 10
RUNS = 5


with open("results-grambank-runs-{0}-{1}-{2}-{3}-{4:.2f}-{5:.2f}-{6}.tsv".format(
    LEVEL,
    min_classes,
    test_size,
    RUNS,
    THRESHOLD,
    tt_split,
    TEST_SIZE), "w") as f:
    f.write("\t".join([
        "Family", 
        "Training",
        "Testing",
        "Samples",
        "Scores",
        "Scores_STD",
        "Positive_Scores",
        "Positive_Scores_STD",
        "Negative_Scores",
        "Negative_Scores_STD",
        "Accuracy",
        "Accuracy_STD"])+"\n")
    for row in table:
        f.write(
                row[0]+
                '\t'+
                '\t'.join([str(x) for x in row[1:4]])+
                '\t'+
                '\t'.join(["{0:.2f}".format(x) for x in row[5:]])+
                "\n")

