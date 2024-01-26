import sqlite3
import lingpy
from collections import defaultdict
import tqdm
from clldutils.misc import slug
import random
import statistics
import numpy as np


ATTACH_BPT = """ATTACH 'data/blumpanotacana.sqlite3' AS db1;"""
ATTACH_IECOR = """ATTACH 'data/iecor.sqlite3' AS db1;"""
ATTACH_VBC = """ATTACH 'data/viegasbarroschaco.sqlite3' AS db1;"""
ATTACH_GBE = """ATTACH 'data/grollemundbantu.sqlite3' AS db1;"""
ATTACH_ASJP = """ATTACH 'data/asjp.sqlite3' AS db1;"""
ATTACH_BC = """ATTACH 'data/birchallchapacuran.sqlite3' AS db1;"""
ATTACH_LB = """ATTACH 'data/lexibank.sqlite3' AS db2;"""

IECOR_QUERY = """
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
      db1.parametertable as p_2a,
      db2.parametertable as p_2
    WHERE
      f_2.cldf_languageReference = l_2.cldf_id
        AND
      lower(p_2a.cldf_name) = p_2.cldf_id
        AND
      f_2.cldf_parameterReference = p_2a.cldf_id
        AND
      (
        p_2.core_concept like "%Swadesh-1952-200%"
          OR
        p_2.core_concept like "%Swadesh-1955-100%"
          OR
        p_2.core_concept like "%Tadmor-2009-100%"
      )
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
  length(f.cldf_segments) > 1
    AND
  c.Word_Number >= 50;
"""

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
  c.Word_Number >= 25;
"""


BPT_QUERY = """
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
      f_2.cldf_parameterReference = p_2.cldf_id
        AND
      (
        p_2.core_concept like "%Swadesh-1952-200%"
          OR
        p_2.core_concept like "%Swadesh-1955-100%"
      )
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
  c.Word_Number >= 50;
"""


LBMOD_QUERY = """
SELECT
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
  db2.formtable as f,
  db2.languagetable as l,
  db2.parametertable as p
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
      p_2.cldf_id = f_2.gloss_in_source
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
  c.Word_Number >= 25;
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
        AND
      (
        p_2.core_concept like "%Swadesh-1952-200%"
          OR
        p_2.core_concept like "%Swadesh-1955-100%"
      )
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
  c.Word_Number >= 50;
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
  parametertable
WHERE
  (
    core_concept like "%Swadesh-1952-200%"
      OR
    core_concept like "%Swadesh-1955-100%"
  );"""


GB_PARAMS = """SELECT
  p.cldf_name, c.cldf_name
FROM
  parametertable as p, codetable as c
WHERE
  p.cldf_id = c.cldf_parameterreference;"""


BRANCH_QUERY = """SELECT
    cldf_languagereference
FROM
    valuetable
WHERE
    cldf_parameterreference = 'classification' and cldf_value like ?
"""


def get_other(mode="bpt"):
    db = get_db("data/dummy.sqlite3")
    wordlists = defaultdict(lambda : defaultdict(dict))
    db.execute(ATTACH_LB)
    if mode == "bpt":
        db.execute(ATTACH_BPT)
        db.execute(BPT_QUERY)
    elif mode == "asjp":
        db.execute(ATTACH_ASJP)
        db.execute(ASJP_QUERY)
    elif mode == "lb_mod":
        db.execute(ATTACH_ASJP)
        db.execute(LBMOD_QUERY)
    elif mode == "iecor":
        db.execute(ATTACH_IECOR)
        db.execute(IECOR_QUERY)
    elif mode == "bc":
        db.execute(ATTACH_BC)
        db.execute(IECOR_QUERY)
    elif mode == "vbc":
        db.execute(ATTACH_VBC)
        db.execute(IECOR_QUERY)

    for idx, lidx, glottocode, family, concept, tokens, cog, size in tqdm.tqdm(db.fetchall()):
        wordlists[glottocode][lidx, size][idx] = [glottocode, family, concept, tokens, lidx, cog]

    # retrieve best glottocodes
    all_wordlists = {}
    for glottocode in wordlists:
        if len(wordlists[glottocode]) == 1:
            best_key = list(wordlists[glottocode].keys())[0]
        else:
            best_key = sorted(
                    wordlists[glottocode].keys(),
                    key=lambda x: x[1],
                    reverse=True)[0]
        all_wordlists[glottocode] = wordlists[glottocode][best_key]
    return all_wordlists


def extract_branch(gcode):
    """
    Retreives all glottocodes that are part of a certain branch
    in Glottolog.
    """
    gcode = "%" + gcode + "%"
    db = get_db("data/glottolog.sqlite3")
    db.execute(BRANCH_QUERY, (gcode,))
    gcodes = []
    for glottocode in db.fetchall():
        gcodes.append(glottocode[0])

    return gcodes


def get_gb(path="data/grambank.sqlite3"):
    """
    Retrieve all wordlists from data.

    Note: fetch biggest by glottocode.
    """
    db = get_db(path)
    wordlists = defaultdict(lambda : defaultdict(dict))
    db.execute(GB_QUERY)
    for idx, glottocode, concept, tokens in tqdm.tqdm(db.fetchall()):
        if tokens:
            wordlists[glottocode][idx] = [
                idx, glottocode, concept, tokens,
                "{0}-{1}".format(slug(concept), tokens)
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
    for _, (param, code) in enumerate(db.fetchall()):
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
    return converter


def concept2vec(db, model="dolgo"):
    """
    Function returns a function that converts data from one language to a vector.

    The representation is based on a certain set of sound classes and the set
    of concepts. For each concept, two slots of n sound classes each are
    provided.
    """
    sc_model = lingpy.rc(model)
    # ugly hack, must refine dolgo-model in lingpy!
    sc_model.tones = "1"

    db.execute(CONCEPT_QUERY)
    concepts = {c[0]: i for i, c in enumerate(db.fetchall())}

    sound_classes = [c for c in
                     sorted(set(sc_model.converter.values())) if c not
                     in "+_" + sc_model.vowels + sc_model.tones] + ["?"]
    cls2idx = {c: i for i, c in enumerate(sound_classes)}

    def converter(words):
        nested_vector = [[len(sound_classes) * [0], len(sound_classes) * [0]] for c in concepts]

        for concept, tokens in words:
            # Addition for BPT to only add parameters that are in lexibank
            # At least one case: MUD
            if concept in concepts:
                class_string = lingpy.tokens2class(tokens, model)
                reduced_string = [t for t in class_string if t in
                                sound_classes][:2]
                if not reduced_string:
                    first, second = "?", "?"
                elif len(reduced_string) == 1:
                    if class_string[0] in sc_model.vowels:
                        first, second = "?", reduced_string[0]
                    else:
                        first, second = reduced_string[0], "?"
                else:
                    first, second = reduced_string
                nested_vector[concepts[concept]][0][cls2idx[first]] = 1
                nested_vector[concepts[concept]][1][cls2idx[second]] = 1
        vector = []
        for a, b in nested_vector:
            vector += a + b
        return vector

    return converter


def get_db(path):
    con = sqlite3.connect(path)
    return con.cursor()


def get_languages(db):
    db.execute("select cldf_id, cldf_name, cldf_glottocode, family from languagetable;")
    return list(db.fetchall())


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
    wordlists = defaultdict(lambda : defaultdict(dict))
    db.execute(WL_QUERY)
    for idx, lidx, glottocode, family, concept, tokens, cog, size in tqdm.tqdm(db.fetchall()):
        wordlists[glottocode][lidx, size][idx] = [glottocode, family, concept, tokens, lidx, cog]

    # retrieve best glottocodeis
    all_wordlists = {}
    for glottocode in wordlists:
        if len(wordlists[glottocode]) == 1:
            best_key = list(wordlists[glottocode].keys())[0]
        else:
            best_key = sorted(
                    wordlists[glottocode].keys(),
                    key=lambda x: x[1],
                    reverse=True)[0]
        all_wordlists[glottocode] = wordlists[glottocode][best_key]
    return all_wordlists


def get_families(wordlists, families, threshold=5):
    """
    Get all families from a wordlist with the family lookup.
    """

    # order by family
    by_fam = defaultdict(lambda : defaultdict(dict))
    for gcode, items in wordlists.items():
        if gcode in families:
            by_fam[families[gcode]][gcode] = items

    return {k: v for k, v in by_fam.items() if len(v) >= threshold}


def convert_data(wordlists, families, converter, load="lexical", threshold=3):
    # order by family
    by_fam = defaultdict(list)
    for gcode in wordlists:
        if gcode in families:
            by_fam[families[gcode]] += [gcode]

        elif load == "tapakuric":
            by_fam["Chapacuran"] += [gcode]

        elif load == "mataguayan":
            by_fam["Mataguayan"] += [gcode]
    # assemble languages belonging to one family alone to form the group of
    # unclassified languages which is our control group (!)
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
    # Convert to vector

    features = []
    labels = []
    languages = []

    all_languages = defaultdict()
    for fam, gcodes in by_fam.items():
        for gcode in gcodes:
            data = wordlists[gcode]
            label = fam2idx[fam]
            if load in ("lexical", "tapakuric", "mataguayan"):
                features = [[row[2], row[3].split()] for row in data.values()]
            if load == "grambank":
                features = [[x[2], x[3]] for x in data.values()]
            feature_vector = converter(features)

            features.append(feature_vector)
            labels.append(label)
            languages.append(gcode)

            all_languages[gcode] = [fam, label, feature_vector]

    return all_languages


def training_data(wordlists, families, sample=0.8, threshold=5):
    """
    Create a train-test split from the wordlists.

    @param wordlists: a nested dictionary.
    @param families: a dictionary with a glottocode as key and a language family as value.
    """
    # order by family
    by_fam = defaultdict(list)
    for gcode in wordlists:
        if gcode in families:
            by_fam[families[gcode]] += [gcode]

    # assemble languages belonging to one family alone to form the group of
    # unclassified languages which is our control group (!)
    unclassified, delis = [], []
    for fam, gcodes in by_fam.items():
        if len(set(gcodes)) == 1:
            unclassified.extend(gcodes)
            delis.append(fam)
    for fam in delis:
        del by_fam[fam]
    by_fam["Unclassified"] = unclassified

    # select 80% of languages per family, retain families with at least 5
    # exemplars
    train, test = {}, {}
    for fam, gcodes in by_fam.items():
        sampled = int(len(gcodes) * sample + 0.5)
        if sampled >= threshold:
            train[fam] = {
                    gcode: wordlists[gcode] for gcode in
                    random.sample(gcodes, sampled)}
            test[fam] = {
                    gcode: wordlists[gcode] for gcode in
                    gcodes if gcode not in train[fam]}
    return train, test


def affiliate_by_grambank(
        language,
        wordlist,
        wordlists,
        criterion="mean"
        ):
    """
    """
    crt = {"mean": 1, "median": 2, "max": 3, "min": 4}

    # transform the language in the lingpy wordlist
    items = set()
    for idx, doculect, concept, tokens in wordlist.iter_rows(
            "doculect", "concept", "tokens"):
        if doculect == language and tokens:
            items.add(
                "{0}-{1}".format(
                slug(concept),
                tokens[0])
                )
    matches = defaultdict(lambda : defaultdict(list))

    classes = []
    for fam, data in wordlists.items():
        scores = []
        for gcode, words in data.items():
            if gcode != language:
                items_b = set([row[5] for row in words.values()])
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
    return sorted(classes, key=lambda x: x[crt[criterion]], reverse=True)[:3]


def affiliate_by_consonant_class(
        language,
        wordlist,
        wordlists,
        criterion="mean"
        ):
    """
    """
    crt = {"mean": 1, "median": 2, "max": 3, "min": 4}

    # transform the language in the lingpy wordlist
    items = set()
    for _, doculect, concept, tokens in wordlist.iter_rows(
            "doculect", "concept", "tokens"):
        if doculect == language and tokens:
            items.add(
                "{0}-{1}".format(
                    slug(concept),
                    "".join(
                        lingpy.tokens2class(
                            tokens, "dolgo")).replace("V", "")[:2])
                        )
    matches = defaultdict(lambda : defaultdict(list))

    classes = []
    for fam, data in wordlists.items():
        scores = []
        for gcode, words in data.items():
            if gcode != language:
                items_b = set([row[5] for row in words.values()])
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
    return sorted(classes, key=lambda x: x[crt[criterion]], reverse=True)[:3]


def get_lingpy(data, header):
    """
    Helper function to get a wordlist (not really needed, for convenience for
    now).
    """
    data[0] = header
    wl = lingpy.Wordlist(data)
    for idx in wl:
        if isinstance(wl[idx, "tokens"], str):
            wl[idx, "tokens"] = wl[idx, "tokens"].split()
    return wl


class FF(object):
    def __init__(
            self,
            input_layer: int,
            hidden_layer: int,
            output_layer: int,
            verbose: bool = False,
            ):
        self.input_layer = np.array(np.random.uniform(
                -1,
                1,
                (input_layer, hidden_layer)),
                                     dtype="longdouble")
        self.output_layer = np.array(np.random.uniform(
                -1,
                1,
                (hidden_layer, output_layer)),
                                     dtype="longdouble")
        self.input_weights = []
        self.output_weights = []
        self.epoch_loss = []
        self.verbose = verbose

    def train(self, data, epochs, learning_rate=0.01):
        for i in range(epochs):
            losses = []
            for input_data, output_data in tqdm.tqdm(
                    data, desc="epoch {0}".format(i+1)):
                # forward pass on the network
                predicted, hidden_layer, _ = self.forward(
                        self.input_layer,
                        self.output_layer,
                        input_data)

                # error calculation
                total_error = self.get_error(predicted, output_data)

                # backward weight adjustment
                self.backward(
                        total_error,
                        hidden_layer,
                        input_data,
                        learning_rate
                        )

                # loss calculation
                losses += [self.get_loss(predicted, output_data)]

            self.epoch_loss.append(statistics.mean(losses))
            self.input_weights.append(self.input_layer)
            self.output_weights.append(self.output_layer)
            if self.verbose:
                print("Epoch: {0}, Loss: {1:.4f}".format(i+1, statistics.mean(losses)))

    def get_error(self, predicted, output_data):
        idxs = set([i for i in range(len(output_data)) if output_data[i] == 1])
        idxs_l = len(idxs)

        total_error = [
                (p - 1) + (idxs_l - 1) * p if i in idxs else idxs_l * p for i, p in enumerate(predicted)
                ]
        return np.array(total_error)

    def get_loss(self, output_layer, output_data):
        """
        Calculate cross-entropy loss.
        """
        return -np.log(sum(np.clip(output_layer, 1e-7, 1 - 1e-7) * output_data))

    def backward(
            self,
            total_error,
            hidden_layer,
            input_data,
            learning_rate
            ):
        dl_hidden_in = np.outer(input_data, np.dot(self.output_layer, total_error.T))
        dl_hidden_out = np.outer(hidden_layer, total_error)

        self.input_layer = self.input_layer - (learning_rate * dl_hidden_in)
        self.output_layer = self.output_layer - (learning_rate * dl_hidden_out)

    def softmax(self, x):
        """
        Following
        https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
        in adding a constant to the softmax calculation to avoid floating point
        or large number problems in numpy.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0, keepdims=True)

    def forward(self, iweights, oweights, ivecs):
        # from input vectors to input weights for first layer
        hidden = np.dot(iweights.T, ivecs)
        # from first layer to output layer
        out = np.dot(oweights.T, hidden)
        # prediction with softmax
        predicted = self.softmax(out)

        return predicted, hidden, out

    def predict(self, x, weights_in, weights_out):

        y, _, _ = self.forward(weights_in, weights_out, x)

        return np.argmax(y)
