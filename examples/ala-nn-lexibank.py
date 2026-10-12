"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, concept2vec, get_db
from ala.utils import write_table, main
from ala.model import run_model, TrainConfig


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    conceptlist='Starostin-1991-110'
    # conceptlist="Swadesh-1955-100"
    # conceptlist="Tadmor-2009-100"
    # conceptlist="Swadesh-1952-200"

    wl, families = load_data('lexibank', conceptlist, args.minimum, args.experiment)

    lb_conv, _ = concept2vec(get_db('data/lexibank.sqlite3'), conceptlist)

    data = convert_data(wl, families, lb_conv, load='lexical')

    isolates = ['bang1363', 'basq1248', 'mapu1245', 'kusu1250', 'cara1273']

    test_langs = {lang: data[lang] for lang in data if (args.experiment and lang in isolates)}

    config = TrainConfig()
    result_per_fam, results_experiment = run_model(data, args.runs, config, test_langs=test_langs)

    # Only write overall results if not runnings the experiment
    if args.experiment is False:
        write_table('lexibank', result_per_fam)

    # Only write 'experiments_' results in the corresponding setting
    if args.experiment:
        write_table('lexibank', results_experiment, experiment=True)


run_ala(main())
