"""IMporting the databases from the different CLDF datasets."""
import sys
import sqlite3
from collections import defaultdict
from lingpy import rc, tokens2class
import tqdm
from clldutils.misc import slug
from soundvectors import SoundVectors
from pyclts import CLTS

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
  c.Word_Number >= 20;
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


WL_QUERY = """
WITH selected_concepts AS (
  SELECT cc.cldf_concepticonReference
  FROM parametertable AS cc
  WHERE cc.core_concept LIKE '%{0}%'
)
SELECT
  ROW_NUMBER() OVER() AS row_num,
  l.cldf_id,
  l.cldf_glottocode,
  l.family,
  p.cldf_name,
  f.cldf_segments,
  p.cldf_id || '-' || SUBSTR(
    REPLACE(REPLACE(REPLACE(f.dolgo_sound_classes, 'V', ''), '+', ''), '1', ''), 0, 3
  ) AS sound_class_id,
  c.word_number
FROM formtable AS f
JOIN languagetable AS l ON f.cldf_languageReference = l.cldf_id
JOIN parametertable AS p ON f.cldf_parameterReference = p.cldf_id
JOIN selected_concepts AS sc ON p.cldf_concepticonReference = sc.cldf_concepticonReference
JOIN (
    SELECT
      l2.cldf_glottocode,
      COUNT(DISTINCT p2.cldf_name) AS word_number
    FROM formtable AS f2
    JOIN languagetable AS l2 ON f2.cldf_languageReference = l2.cldf_id
    JOIN parametertable AS p2 ON f2.cldf_parameterReference = p2.cldf_id
    JOIN selected_concepts AS sc2 ON p2.cldf_concepticonReference = sc2.cldf_concepticonReference
    GROUP BY l2.cldf_glottocode
) AS c ON c.cldf_glottocode = l.cldf_glottocode
WHERE c.word_number >= 50
;
"""


GB_QUERY = """SELECT
  ROW_NUMBER() OVER(),
  l.cldf_id,
  p.cldf_name,
  f.cldf_value,
  l.Family_name
FROM
  valuetable as f,
  languagetable as l,
  parametertable as p
ON
  f.cldf_languagereference = l.cldf_id
WHERE
  f.cldf_parameterReference = p.cldf_id;
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
  p.cldf_id = c.cldf_parameterreference;
"""


def _get_best_key(wordlists, glottocode):
    if len(wordlists[glottocode]) == 1:
        return list(wordlists[glottocode].keys())[0]

    return sorted(wordlists[glottocode].keys(), key=lambda x: x[1], reverse=True)[0]


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
        best_key = _get_best_key(wordlists, glottocode)
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
    for idx, glottocode, concept, tokens, family in tqdm.tqdm(db.fetchall()):
        if tokens:
            wordlists[glottocode][idx] = [
                glottocode, family, concept, tokens, f"{slug(concept)}-{tokens}"
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


def concept2featurevec(db, conceptlist, n_consonants=4):
    """
    Function returns a function that converts data from one language to a vector.

    Instead of one-hot sound-class vectors (as in concept2vec), this uses
    phonetic feature vectors from the `soundvectors` package, computed via a
    CLTS transcription system (e.g. BIPA). For each concept, `n_consonants`
    slots are reserved, each filled with the feature vector of the
    corresponding consonant segment. Missing/short words are padded with a
    neutral (zero) vector.
    """
    ts = CLTS().bipa
    sv = SoundVectors(ts=ts)

    db.execute(CONCEPT_QUERY.format(conceptlist))
    concepts = {c[0]: i for i, c in enumerate(db.fetchall())}

    empty_vec = list(tuple([0] * len(sv.get_vec("b"))))[1:25]

    def converter(words):
        nested_vector = [[list(empty_vec) for _ in range(n_consonants)] for _ in concepts]

        for concept, segments in words:
            count = 0
            idx = concepts[concept]
            for segment in segments:
                if count >= n_consonants:
                    break
                # Check if valid segment
                try:
                    feat_vec = list(sv.get_vec(segment))
                except (KeyError, ValueError):
                    continue

                if feat_vec[0] == 1:
                    nested_vector[idx][count] = feat_vec[1:25]
                    count += 1
            # print(segments, nested_vector[idx])
        vector = [x for group in nested_vector for sub in group for x in sub]

        return vector

    return converter


def concept2vec(db, conceptlist, model="dolgo", h_class="H"):
    """
    Function returns a function that converts data from one language to a vector.

    The representation is based on a certain set of sound classes and the set
    of concepts. For each concept, two slots of n sound classes each are
    provided.
    """
    sc_model = rc(model)

    # ugly hack, must refine dolgo-model in lingpy!
    sc_model.tones = "1"

    db.execute(CONCEPT_QUERY.format(conceptlist))
    concepts = {c[0]: i for i, c in enumerate(db.fetchall())}

    sound_classes = [c for c in
                     sorted(set(sc_model.converter.values())) if c not
                     in "+_" + sc_model.vowels + sc_model.tones] + ["?"]
    cls2idx = {c: i for i, c in enumerate(sound_classes)}

    def converter(words):
        # modify so that I can switch between 1-3 consonants dynamically
        n_consonants = 2

        nested_vector = [[len(sound_classes) * [0] for _ in range(n_consonants)] for c in concepts]

        for concept, tokens in words:
            class_string = tokens2class(tokens, model)
            reduced_string = [t for t in class_string if t in sound_classes][:n_consonants]

            idx = concepts[concept]
            for pos in range(n_consonants):
                cls = reduced_string[pos] if pos < len(reduced_string) else h_class
                nested_vector[idx][pos][cls2idx[cls]] = 1

        vector = [x for group in nested_vector for sub in group for x in sub]

        return vector

    return converter


def get_db(path):
    """Simple access to a SQLite3 db."""
    con = sqlite3.connect(path)
    return con.cursor()


def get_asjp(path="data/asjp.sqlite3"):
    """Currently not used anymore. Could be reused for WALS classification if needed"""

    db = get_db(path)
    db.execute("select distinct cldf_glottocode, family, classification_wals from languagetable;")
    return {a: [b, c] for a, b, c in db.fetchall()}


def get_lb(conceptlist, path="data/lexibank.sqlite3"):
    """
    Retrieve all wordlists from data.
    Note: fetch biggest by glottocode.
    """
    db = get_db(path)
    wl = defaultdict(lambda: defaultdict(dict))
    db.execute(WL_QUERY.format(conceptlist))
    for idx, lidx, glottocode, family, concept, tokens, cog, size in tqdm.tqdm(db.fetchall()):
        wl[glottocode][lidx, size][idx] = [glottocode, family, concept, tokens, lidx, cog]

    # retrieve best glottocodes
    all_wordlists = {glottocode: wl[glottocode][_get_best_key(wl, glottocode)] for glottocode in wl}
    return all_wordlists


def convert_data(wordlists, families, converter, load="lexical"):
    """Converts data into one-hot vectors."""
    fam2idx = {fam: i for i, fam in enumerate(families)}
    all_languages = defaultdict()

    for fam, gcodes in families.items():
        for gcode in gcodes:
            data = wordlists[gcode]
            label = fam2idx[fam]
            if load == "lexical":
                features = [[row[2], row[3].split()] for row in data.values()]
            elif load == "grambank":
                features = [[x[2], x[3]] for x in data.values()]
            else:
                raise ValueError(f"Unknown database: {load}")

            feature_vector = converter(features)

            all_languages[gcode] = [fam, label, feature_vector]

    return all_languages


def load_data(database, threshold, experiment=False, conceptlist='Swadesh-1955-100'):
    """
    Loads the datasets and selects a subset.
    This includes filtering the dataset for our threshold.
    """
    gb = get_gb()
    lb = get_lb(conceptlist)
    # asjp_data = get_other(mode="asjp")

    # Get intersection of datasets
    # if experiment is False:
    #     common_languages = [lng for lng in lb if lng in gb] # and lng in asjp_data]

    #     gb = {k: v for k, v in gb.items() if k in common_languages}
    #     lb = {k: v for k, v in lb.items() if k in common_languages}

    data = lb if database == 'lexibank' else gb if database == 'grambank' else None #asjp_data

    # get number of families to be inferred
    language_families = defaultdict(int)
    for _, wl in data.items():
        language_families[list(wl.values())[0][1]] += 1

    selected_fams = {fam: num for fam, num in language_families.items() if num >= threshold}

    selected_langs = {}
    by_fam = defaultdict(list)

    for gcode, wl in data.items():
        fam = list(wl.values())[0][1]
        # TODO: try language_families[fam] < threshold
        if fam is None or language_families[fam] == 1:
            selected_langs[gcode] = 'Unclassified'
            by_fam['Unclassified'].append(gcode)

        elif fam in selected_fams:
            selected_langs[gcode] = fam
            by_fam[fam].append(gcode)

    # Summary stats
    print(f'This run includes {len(selected_langs)} languages from {len(selected_fams)} families.')

    # Load main data
    data_map = {'lexibank': lb, 'grambank': gb, 'combined': lb}
    wl = data_map[database]

    if database not in data_map:
        print("Invalid data selection. Please choose 'lexibank', 'grambank', 'combined' or 'asjp'")
        sys.exit()

    return wl, by_fam
