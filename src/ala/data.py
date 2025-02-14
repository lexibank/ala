import sys
import sqlite3
from collections import defaultdict
from lingpy import rc, tokens2class
import tqdm
from clldutils.misc import slug


ATTACH_ASJP = """ATTACH '{0}/asjp.sqlite3' AS db1;"""
ATTACH_CAR = """ATTACH '{0}/carari.sqlite3' AS db1;"""
ATTACH_LB = """ATTACH '{0}/lexibank.sqlite3' AS db2;"""

ASJP_QUERY = """
SELECT
  ROW_NUMBER() OVER(),
  l.cldf_id,
  l.cldf_glottocode,
  l.family,
  p.concepticon_gloss,
  f.cldf_segments,
  p.cldf_id,
  c.Word_Number
FROM
  db1.formtable AS f,
  db1.languagetable AS l,
  db1.parametertable AS p
INNER JOIN
  (
    SELECT
      l_2.cldf_glottocode,
      COUNT (*) as Word_Number
    FROM
      db1.formtable as f_2,
      db1.languagetable as l_2,
      db2.parametertable as p_2
    WHERE
      f_2.cldf_languageReference = l_2.cldf_id
        AND
      f_2.gloss_in_source = p_2.cldf_id
        AND
      p_2.core_concept like "%Holman-2008-40%"
    GROUP BY
      l_2.cldf_glottocode
  ) as c
ON
  c.cldf_glottocode = l.cldf_glottocode
WHERE
  f.cldf_parameterReference = p.cldf_id
    AND
  f.cldf_languageReference = l.cldf_id
    AND
  c.Word_Number >= 30;
"""

CAR_QUERY = """
SELECT
  ROW_NUMBER() OVER(),
  l.cldf_id,
  l.cldf_glottocode,
  l.family,
  p.concepticon_gloss,
  f.cldf_segments,
  p.cldf_id,
  c.word_number
FROM
  db1.formtable AS f,
  db1.languagetable AS l,
  db1.parametertable AS p
INNER JOIN
  (
    SELECT
      l_2.cldf_glottocode,
      COUNT (*) as Word_Number
    FROM
      db1.formtable as f_2,
      db1.languagetable as l_2,
      db1.parametertable as p_2,
      db2.parametertable as cc
    WHERE
      f_2.cldf_languageReference = l_2.cldf_id
        AND
      f_2.cldf_parameterReference = p_2.cldf_id
        AND
      p_2.cldf_concepticonReference = cc.cldf_concepticonReference
        AND
      cc.core_concept like '%Tadmor-2009-100%'
    GROUP BY
      l_2.cldf_glottocode
  ) as c
ON
  c.cldf_glottocode = l.cldf_glottocode
WHERE
  f.cldf_parameterReference = p.cldf_id
    AND
  f.cldf_languageReference = l.cldf_id
    AND
  l.cldf_glottocode = 'cara1273'
;
"""

WL_QUERY = """SELECT
  ROW_NUMBER() OVER(),
  l.cldf_id,
  l.cldf_glottocode,
  l.family,
  p.cldf_name,
  f.cldf_segments,
  p.cldf_id || "-" || SUBSTR(
    REPLACE(REPLACE(REPLACE(f.dolgo_sound_classes, "V", ""), "+", ""), "1", ""), 0, 3),
  c.Word_Number
FROM
  formtable as f,
  languagetable as l,
  parametertable as p
INNER JOIN
  (
    SELECT
      l_2.cldf_glottocode,
      COUNT (*) as Word_Number
    FROM
      formtable as f_2,
      languagetable as l_2,
      parametertable as p_2
    WHERE
      f_2.cldf_languageReference = l_2.cldf_id
        AND
      f_2.cldf_parameterReference = p_2.cldf_id
    GROUP BY
      l_2.cldf_glottocode
  ) as c
ON
  c.cldf_glottocode = l.cldf_glottocode
WHERE
  f.cldf_parameterReference = p.cldf_id
    AND
  f.cldf_languageReference = l.cldf_id
    AND
  c.Word_Number >= 40
;
"""


GB_QUERY = """SELECT
  ROW_NUMBER() OVER(),
  l.cldf_id,
  p.cldf_name,
  f.cldf_value
FROM
  valuetable as f,
  languagetable as l,
  parametertable as p
ON
  f.cldf_languagereference = l.cldf_id
WHERE
  f.cldf_parameterReference = p.cldf_id
    AND
  f.cldf_languageReference = l.cldf_id;
"""


CONCEPT_QUERY = """SELECT
  cldf_name
FROM
  parametertable as p
WHERE
  p.core_concept like "%{0}%"
;
"""


GB_PARAMS = """SELECT
  p.cldf_name, c.cldf_name
FROM
  parametertable as p, codetable as c
WHERE
  p.cldf_id = c.cldf_parameterreference;"""


def get_best_key(wordlists, glottocode):
    if len(wordlists[glottocode]) == 1:
        return list(wordlists[glottocode].keys())[0]
    return sorted(
        wordlists[glottocode].keys(),
        key=lambda x: x[1],
        reverse=True)[0]


def get_other(mode, data_path=None):
    data_path = "data" if not data_path else data_path
    db = get_db("")
    wordlists = defaultdict(lambda: defaultdict(dict))
    db.execute(ATTACH_LB.format(data_path))
    attach_queries = {
        "asjp": ATTACH_ASJP.format(data_path),
        "lb_mod": ATTACH_ASJP.format(data_path),
        "carari": ATTACH_CAR.format(data_path)
    }
    select_queries = {
        "asjp": ASJP_QUERY.format(data_path),
        "carari": CAR_QUERY.format(data_path)
    }
    db.execute(attach_queries[mode])
    db.execute(select_queries[mode])

    for idx, lidx, glottocode, family, concept, tokens, cog, size in tqdm.tqdm(db.fetchall()):
        wordlists[glottocode][lidx, size][idx] = [glottocode, family, concept, tokens, lidx, cog]

    # retrieve best glottocodes
    all_wordlists = {}
    for glottocode in wordlists:
        best_key = get_best_key(wordlists, glottocode)
        all_wordlists[glottocode] = wordlists[glottocode][best_key]

    return all_wordlists


def get_gb(path="data/grambank.sqlite3"):
    """
    Retrieve all wordlists from data.

    Note: fetch biggest by glottocode.
    """
    db = get_db(path)
    wordlists = defaultdict(lambda: defaultdict(dict))
    db.execute(GB_QUERY)
    for idx, glottocode, concept, tokens in tqdm.tqdm(db.fetchall()):
        if tokens:
            wordlists[glottocode][idx] = [
                idx, glottocode, concept,
                tokens, f"{slug(concept)}-{tokens}"
                ]
    return wordlists


def feature2vec(db):
    """
    Function turns data from one language into a flat vector.
    """
    db.execute(GB_PARAMS)
    # we need to find out for each param, how many values it has, so we do a
    # query on grambank here
    keys = defaultdict(dict)
    idx = 0
    for param, code in db.fetchall():
        if code != "3":
            keys[param][code] = idx
            idx += 1

    # Iterate over the data, passed as pairs of parameter and value
    def converter(words):
        vector = [0 for x in range(idx+1)]
        for param, value in words:
            if value == str(3):
                vector[keys[param][str(1)]] = 1
                vector[keys[param][str(2)]] = 1
            else:
                vector[keys[param][value]] = 1
        return vector

    return converter, len(keys)


def concept2vec(db, model="dolgo", h_class="H", conceptlist="Swadesh-1955-100"):
    """
    Function returns a function that converts data from one language to a vector.

    The representation is based on a certain set of sound classes and the set
    of concepts. For each concept, two slots of n sound classes each are
    provided.
    """
    sc_model = rc(model)

    # ugly hack, must refine dolgo-model in lingpy!
    if model == "dolgo":
        sc_model.tones = "1"

    db.execute(CONCEPT_QUERY.format(conceptlist))
    concepts = {c[0]: i for i, c in enumerate(db.fetchall())}

    sound_classes = [c for c in
                     sorted(set(sc_model.converter.values())) if c not
                     in "+_" + sc_model.vowels + sc_model.tones] + ["?"]
    cls2idx = {c: i for i, c in enumerate(sound_classes)}

    def converter(words):
        nested_vector = [[len(sound_classes) * [0], len(sound_classes) * [0]] for c in concepts]
        for concept, tokens in words:
            if concept in concepts:
                class_string = tokens2class(tokens, model)
                reduced_string = [t for t in class_string if t in sound_classes][:2]
                first = h_class if len(reduced_string) == 0 else reduced_string[0]
                second = h_class if len(reduced_string) < 2 else reduced_string[1]

                idx = concepts[concept]
                nested_vector[idx][0][cls2idx[first]] = 1
                nested_vector[idx][1][cls2idx[second]] = 1

        vector = [x for a, b in nested_vector for x in a + b]

        return vector

    return converter, 2 * len(sound_classes)


def get_db(path):
    con = sqlite3.connect(path)
    return con.cursor()


def get_asjp(path="data/asjp.sqlite3"):
    db = get_db(path)
    db.execute("select distinct cldf_glottocode, family, classification_wals from languagetable;")
    return {a: [b, c] for a, b, c in db.fetchall()}


def get_lb(path="data/lexibank.sqlite3"):
    """
    Retrieve all wordlists from data.

    Note: fetch biggest by glottocode.
    """
    db = get_db(path)
    wl = defaultdict(lambda: defaultdict(dict))
    db.execute(WL_QUERY)
    for idx, lidx, glottocode, family, concept, tokens, cog, size in tqdm.tqdm(db.fetchall()):
        wl[glottocode][lidx, size][idx] = [glottocode, family, concept, tokens, lidx, cog]

    # retrieve best glottocodes
    all_wordlists = {glottocode: wl[glottocode][get_best_key(wl, glottocode)] for glottocode in wl}
    return all_wordlists


def convert_data(wordlists, families, converter, load="lexical", threshold=3):
    # order by family
    by_fam = defaultdict(list)
    for gcode in wordlists:
        if gcode in families:
            by_fam[families[gcode]] += [gcode]

    unclassified, delis = [], []
    for fam, gcodes in by_fam.items():
        if len(set(gcodes)) == 1:
            unclassified.extend(gcodes)
            delis.append(fam)
        elif len(set(gcodes)) < threshold:
            delis.append(fam)
    for fam in delis:
        del by_fam[fam]
    by_fam["Unclassified"] = unclassified

    fam2idx = {fam: i for i, fam in enumerate(by_fam)}

    features = []
    labels = []
    languages = []

    all_languages = defaultdict()
    for fam, gcodes in by_fam.items():
        for gcode in gcodes:
            data = wordlists[gcode]
            label = fam2idx[fam]
            if load in ("lexical"):
                features = [[row[2], row[3].split()] for row in data.values()]
            if load == "grambank":
                features = [[x[2], x[3]] for x in data.values()]
            feature_vector = converter(features)

            features.append(feature_vector)
            labels.append(label)
            languages.append(gcode)

            all_languages[gcode] = [fam, label, feature_vector]

    return all_languages


def load_data(database, threshold, experiment=False):
    """
    Loads the datasets and selects a subset.
    """
    # Setup for databases
    gb = get_gb()
    lb = get_lb()
    asjp_data = get_other(mode="asjp")

    if experiment is False:
        # prepare datasets, only use common languages
        common_languages = [lng for lng in asjp_data if lng in gb and lng in lb]

        asjp_data = {k: v for k, v in asjp_data.items() if k in common_languages}
        gb = {k: v for k, v in gb.items() if k in common_languages}
        lb = {k: v for k, v in lb.items() if k in common_languages}

    # get number of families to be inferred
    language_families = defaultdict(int)
    for _, wl in lb.items():
        language_families[list(wl.values())[0][1]] += 1

    selected_families = {fam for fam, num in language_families.items() if num >= threshold}
    selected_languages = [k for k, wl in lb.items() if list(wl.values())[0][1] in selected_families]

    # Summary stats
    print(f'This run includes {len(selected_languages)} languages from {len(selected_families)} families.')

    # Load main data
    data_map = {'lexibank': lb, 'grambank': gb, 'asjp': asjp_data, 'combined': lb}
    wl = data_map[database]

    if database not in data_map:
        print("Invalid data selection. Please choose 'lexibank', 'grambank', 'combined' or 'asjp'")
        sys.exit()

    return wl
