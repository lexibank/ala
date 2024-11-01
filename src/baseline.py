from collections import defaultdict
import random
import statistics
import numpy as np
import lingpy
import tqdm
from clldutils.misc import slug


def get_languages(db):
    db.execute("select cldf_id, cldf_name, cldf_glottocode, family from languagetable;")
    return list(db.fetchall())


def get_families(wordlists, families, threshold=5):
    """
    Get all families from a wordlist with the family lookup.
    """

    # order by family
    by_fam = defaultdict(lambda: defaultdict(dict))
    for gcode, items in wordlists.items():
        if gcode in families:
            by_fam[families[gcode]][gcode] = items

    return {k: v for k, v in by_fam.items() if len(v) >= threshold}


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
    matches = defaultdict(lambda: defaultdict(list))

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
    matches = defaultdict(lambda: defaultdict(list))

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
