"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, concept2featurevec, get_db
from ala.utils import main, plot_grid_heatmap
from ala.model import sweep_grid, TrainConfig


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    conceptlist='Starostin-1991-110'

    wl, families = load_data('lexibank', args.minimum, args.experiment, conceptlist)
    lb_conv = concept2featurevec(get_db('data/lexibank.sqlite3'), conceptlist)
    data = convert_data(wl, families, lb_conv, load='lexical')

    config = TrainConfig()
    param_grid = {
        'lr': [0.0005, 0.0001, 0.00005],
        'batch': [8, 16, 32],
        'hidden': [3]
    }

    sweep_results, param_names = sweep_grid(data, config, param_grid, runs=5)

    best_combo = max(sweep_results.values(), key=lambda r: r['avg_acc'])
    print('Best combination:', best_combo['params'], 'avg_acc:', best_combo['avg_acc'])

    plot_grid_heatmap(sweep_results, param_names, out='results/grid_sweep.png')


run_ala(main())
