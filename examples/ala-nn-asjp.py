"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, concept2vec, get_db, get_asjp
from ala.utils import write_table, main
from ala.model import train


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    wl = load_data('asjp', args.minimum, args.experiment)
    asjp = get_asjp()

    asjp_conv, _ = concept2vec(
            get_db('data/lexibank.sqlite3'), model='dolgo', conceptlist="Holman-2008-40")

    data = convert_data(wl, {k: v[0] for k, v in asjp.items()},
                        asjp_conv, load='lexical', threshold=args.minimum)

    test_langs = []
    result_per_fam, _ = train(
        data, args.runs, test_langs=test_langs, experiment=args.experiment)

    write_table('asjp', result_per_fam, experiment=args.experiment, print_table=True)

run_ala(main())
