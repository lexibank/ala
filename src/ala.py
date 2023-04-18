import sqlite3
import lingpy
from collections import defaultdict
import tqdm
from clldutils.misc import slug
import random
import statistics


WL_QUERY = """SELECT 
  ROW_NUMBER() OVER(),
  l.cldf_id,
  l.cldf_glottocode, 
  l.family,
  p.cldf_name, 
  f.cldf_segments, 
  p.cldf_id || "-" || SUBSTR(
    REPLACE(f.dolgo_sound_classes, "V", ""), 0, 3),
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
  c.Word_Number >= 100
    AND
  (
    p.core_concept like "%Swadesh-1952-200%"
      OR
    p.core_concept like "%Swadesh-1955-100%"
  )
;"""


WL_QUERY2 = """SELECT 
  ROW_NUMBER() OVER() as ID,
  l.cldf_id as IDX,
  l.cldf_glottocode as DOCULECT, 
  p.cldf_name as CONCEPT, 
  f.cldf_segments as TOKENS, 
  p.cldf_id || "-" || SUBSTR(
    REPLACE(f.dolgo_sound_classes, "V", ""), 0, 3) as COG
FROM
  formtable as f,
  languagetable as l,
  parametertable as p
WHERE
  f.cldf_parameterReference = p.cldf_id
    AND
  f.cldf_languageReference = l.cldf_id
    AND
  (
    p.core_concept like "%Swadesh-1952-200%"
      OR
    p.core_concept like "%Swadesh-1955-100%"
  )
;"""


def get_db(path):
    con = sqlite3.connect(path)
    return con.cursor()


def get_languages(db):
    db.execute("select cldf_id, cldf_name, cldf_glottocode, family from languagetable;")
    return list(db.fetchall())


def get_asjp(path="asjp.sqlite3"):
    db = get_db(path)
    db.execute("select distinct cldf_glottocode, family, classification_wals from languagetable;")
    return {a: [b, c] for a, b, c in db.fetchall()}


def get_wordlists(path="lexibank.sqlite3"):
    """
    Retrieve all wordlists from data.

    Note: fetch biggest by glottocode.
    """
    db = get_db(path)
    wordlists = defaultdict(lambda : defaultdict(dict))
    families = {row[2]: row[3] for row in get_languages(db)}
    db.execute(WL_QUERY)
    for idx, lidx, glottocode, family, concept, tokens, cog, size in tqdm.tqdm(db.fetchall()):
        wordlists[glottocode][lidx, size][idx] = [lidx, glottocode, family, concept, tokens, cog]
        
    # retrieve best glottocodeis
    all_wordlists = {}
    for glottocode in wordlists:
        if len(wordlists[glottocode]) == 1:
            best_key = list(wordlists[glottocode].keys())[0]
        else:
            best_key = sorted(
                    wordlist[glottocode].keys(),
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
    for idx, doculect, concept, tokens in wordlist.iter_rows(
            "doculect", "concept", "tokens"):
        if doculect == language:
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
                items_b =  set([row[5] for row in words.values()])
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
        wl[idx, "tokens"] = wl[idx, "tokens"].split()
    return wl


