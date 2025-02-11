"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, concept2vec, get_db, get_asjp
from ala.utils import write_table, main
from ala.model import train


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    wl = load_data('lexibank', args.minimum, args.intersection)
    asjp = get_asjp()

    lb_conv = concept2vec(get_db('data/lexibank.sqlite3'), model='dolgo',
                          conceptlist="Swadesh-1955-100")

    data = convert_data(wl, {k: v[0] for k, v in asjp.items()},
                        lb_conv, load='lexical', threshold=args.minimum)

    result_per_fam = train(data, args.runs)

    write_table('lexibank', result_per_fam, intersection=args.intersection, print_table=True)


run_ala(main())
