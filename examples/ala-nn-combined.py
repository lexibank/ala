"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, feature2vec, concept2vec, get_db, get_other
from ala.utils import write_table, main
from ala.model import run_model, TrainConfig


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    conceptlist='Starostin-1991-110'

    # get configuration options
    wl, families = load_data('lexibank', args.minimum, args.experiment, conceptlist, small_fams=[])
    gb, families = load_data('grambank', args.minimum, args.experiment)

    gb_conv = feature2vec(get_db('data/grambank.sqlite3'))
    lb_conv = concept2vec(get_db('data/lexibank.sqlite3'), conceptlist)

    # Set up combination of LB and GB
    data = convert_data(wl, families, lb_conv, load='lexical')
    gb_wl = convert_data(gb, families, gb_conv, load='grambank')

    # Combine data vectors
    for lang in data:
        data[lang][2] += gb_wl[lang][2]

    config = TrainConfig(lr=1e-3)
    result_per_fam, results_experiment = run_model(data, args.runs, config)

    # Only write overall results if not runnings the experiment
    if args.experiment is False:
        write_table('combined', result_per_fam)

run_ala(main())
