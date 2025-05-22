from lingpy import rc, tokens2class
import statistics
from collections import defaultdict


def affiliate_by_consonant_class(
        language,
        wordlist,
        wordlists,
        families=None
        ):
    """
    """

    crt = {"mean": 1, "median": 2, "max": 3, "min": 4}

    # transform the language in the lingpy wordlist
    items = set()
    for row in wordlist.values():
        items.add(row[-1])
    matches = defaultdict(lambda: defaultdict(list))

    by_fam = defaultdict(dict)
    for gcode, wl in wordlists.items():
        fam = list(wl.values())[0][1]
        by_fam[fam][gcode] = wl 
    families = families or sorted(by_fam)

    classes = []
    for fam, data in by_fam.items():
        scores = []
        for gcode, words in data.items():
            if gcode != language:
                items_b = set([row[-1] for row in words.values()])
                commons = items.intersection(items_b)
                matches = len(commons) / len(items)
                scores += [matches]

        classes += [(
            fam,
            statistics.mean(scores),
            statistics.median(scores),
            max(scores),
            min(scores)
            )]
        print(scores)
    results = [
            sorted(classes, key=lambda x: x[crt["mean"]], reverse=True)[0],
            sorted(classes, key=lambda x: x[crt["median"]], reverse=True)[0],
            sorted(classes, key=lambda x: x[crt["max"]], reverse=True)[0],
            sorted(classes, key=lambda x: x[crt["min"]], reverse=True)[0]
            ]

    return results  #sorted(classes, key=lambda x: x[crt[criterion]], reverse=True)


def get_sound_class(concept, tokens, model="dolgo"):
    sc_model = rc(model)
    # ugly hack, must refine dolgo-model in lingpy!
    sc_model.tones = "1"

    sound_classes = [c for c in
                     sorted(set(sc_model.converter.values())) if c not
                     in "+_" + sc_model.vowels + sc_model.tones] + ["?"]

    class_string = tokens2class(tokens.split(), model)
    reduced_string = [t for t in class_string if t in sound_classes][:2]
    first = "H" if len(reduced_string) == 0 else reduced_string[0]
    second = "H" if len(reduced_string) < 2 else reduced_string[1]
    return concept + "-" + first + second
