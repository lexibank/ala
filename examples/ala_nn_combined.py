"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, concept2vec, feature2vec, get_db, get_asjp
from ala.utils import write_table, main
from ala.model import train


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    wl = load_data('lexibank', args.minimum, args.intersection)
    gb = load_data('grambank', args.minimum, args.intersection)
    asjp = get_asjp()

    gb_conv = feature2vec(get_db('data/grambank.sqlite3'))
    lb_conv = concept2vec(
        get_db('data/lexibank.sqlite3'), model='dolgo', conceptlist="Swadesh-1955-100")

    # Set up combination of LB and GB
    data = convert_data(wl, {k: v[0] for k, v in asjp.items() if k in gb},
                        lb_conv, load='lexical', threshold=args.minimum)
    gb_wl = convert_data(gb, {k: v[0] for k, v in asjp.items()},
                         gb_conv, load='grambank', threshold=args.minimum)

    # Combine data vectors
    for lang in data:
        data[lang][2] += gb_wl[lang][2]

    result_per_fam = train(data, args.runs)

    write_table('combined', result_per_fam, intersection=args.intersection,  print_table=True)


run_ala(main())
