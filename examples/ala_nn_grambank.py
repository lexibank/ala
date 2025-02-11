"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, feature2vec, get_db, get_asjp
from ala.utils import write_table, main
from ala.model import train


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    gb = load_data('grambank', args.minimum, args.intersection)
    asjp = get_asjp()

    gb_conv = feature2vec(get_db('data/grambank.sqlite3'))
    data = convert_data(gb, {k: v[0] for k, v in asjp.items()},
                        gb_conv, load='grambank', threshold=args.minimum)

    result_per_fam = train(data, args.runs)

    write_table('grambank', result_per_fam, intersection=args.intersection, print_table=True)


run_ala(main())
