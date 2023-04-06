from ala import get_wordlists, affiliate_by_consonant_class, get_asjp, training_data, get_lingpy
import random
import statistics

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
for fam, data in test.items():
    selected = random.sample(
            [gcode for gcode in data], 
            5 if 5 <= len(data) else len(data)
            )
    print(fam, len(selected))
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
        best_fam_score = fams[0][1]
        if fam == best_fam:
            hits += [1]
            scores += [best_fam_score]
        else:
            hits += [0]
            scores += [best_fam_score]
print(len(hits))
print(sum(hits)/len(hits))
print(statistics.mean(scores))
print(statistics.mean([s for h, s in zip(hits, scores) if h == 1]))
print(statistics.mean([s for h, s in zip(hits, scores) if h == 0]))

