"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, concept2vec, get_db
from ala.utils import main, plot_grid_heatmap
from ala.model import sweep_grid, TrainConfig


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    wl, families = load_data('lexibank', args.minimum, args.experiment)
    lb_conv, _ = concept2vec(
        get_db('data/lexibank.sqlite3'), model='dolgo',
        conceptlist="Starostin-1991-110"
        )

    data = convert_data(wl, families, lb_conv, load='lexical')

    param_grid = {
        'lr': [1e-2, 5e-3, 3e-3, 1e-3, 5e-4],
        'batch': [8, 16, 32, 64],
        'hidden': [3, 4, 5],
    }

    config = TrainConfig(runs=10)
    sweep_results, param_names = sweep_grid(data, param_grid, config)

    best_combo = max(sweep_results.values(), key=lambda r: r['avg_acc'])
    print('Best combination:', best_combo['params'], 'avg_acc:', best_combo['avg_acc'])

    plot_grid_heatmap(sweep_results, param_names, out_path='results/grid_sweep.png')


run_ala(main())
