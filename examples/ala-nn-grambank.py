"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, feature2vec, get_db
from ala.utils import write_table, main, extract_branch
from ala.model import run_model, TrainConfig


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    gb, families = load_data('grambank', args.minimum, args.experiment)

    gb_conv = feature2vec(get_db('data/grambank.sqlite3'))
    data = convert_data(gb, families, gb_conv, load='grambank')

    # isolates = ['bang1363', 'basq1248', 'mapu1245', 'kusu1250', 'cara1273']

    config = TrainConfig(batch=8, lr=5e-3)  # default batch is 8
    result_per_fam, results_experiment = run_model(data, args.runs, config)

    # Only write overall results if not runnings the experiment
    if args.experiment is False:
        write_table('grambank', result_per_fam)

    # Only write 'experiments_' results in the corresponding setting
    if args.experiment:
        write_table('grambank', results_experiment, experiment=True)


run_ala(main())
