"""Utility functions for the ALA project."""
import argparse
import csv
import os
from collections import Counter
from statistics import mean, stdev
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from ala.data import get_db


BRANCH_QUERY = """SELECT
    cldf_languagereference
FROM
    valuetable
WHERE
    cldf_parameterreference = 'classification' and cldf_value like ?
"""


def extract_branch(gcode):
    """
    Retrieves all glottocodes that are part of a certain branch in Glottolog.
    """
    gcode = "%" + gcode + "%"
    db = get_db("data/glottolog.sqlite3")
    db.execute(BRANCH_QUERY, (gcode,))
    gcodes = [glottocode[0] for glottocode in db.fetchall()]

    return gcodes


def _write_tsv(path, header, rows):
    with open(path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)


def _print_summary_table(results):
    table = [
        [
            fam,
            mean(r[2] for r in rows),
            round(mean(r[3] for r in rows), 1),
            round(mean(r[4] for r in rows), 2),
            round(stdev(r[4] for r in rows), 2) if len(rows) >= 2 else 0,
        ]
        for fam, rows in results.items()
    ]
    table.sort(key=lambda row: (row[0] == 'TOTAL', row[0]))

    headers = ['Family', 'Languages', 'Tested', 'Avg. Fam. Accuracy', 'Fam-STD']
    print(tabulate(table, headers=headers, tablefmt="grid", floatfmt=".2f"))


def write_table(database, results, experiment=False, print_table=True):
    """
    Writes the results to a file and optionally prints a summary table.
    """
    if experiment:
        rows = [
            [lang, fam, pred, freq]
            for (lang, fam), values in results.items()
            for pred, freq in Counter(values).items()
        ]
        output = os.path.join('results', 'experiments_' + database + '.tsv')
        _write_tsv(output, ['Language', 'Family', 'Prediction', 'Frequency'], rows)

    else:
        rows = [row for fam, fam_rows in results.items() if fam != 'TOTAL' for row in fam_rows]
        output = os.path.join('results', 'results_' + database + '.tsv')
        _write_tsv(output, ['Run', 'Family', 'Languages', 'Tested', 'Score'], rows)


    if print_table:
        _print_summary_table(results)


def plot_grid_heatmap(sweep_results, param_names, out=None):
    """Plots the results of the hyperparameter tests."""
    assert len(param_names) == 3, "This plotting function expects exactly three hyperparameters"
    p1, p2, p3 = param_names

    p1_vals = sorted({r['params'][p1] for r in sweep_results.values()})
    p2_vals = sorted({r['params'][p2] for r in sweep_results.values()})
    p3_vals = sorted({r['params'][p3] for r in sweep_results.values()})

    fig, axes = plt.subplots(1, len(p3_vals), figsize=(6 * len(p3_vals), 5), squeeze=False)
    axes = axes[0]

    for ax, p3_val in zip(axes, p3_vals):
        grid = np.full((len(p2_vals), len(p1_vals)), np.nan)

        for r in sweep_results.values():
            params = r['params']
            if params[p3] == p3_val:
                i = p2_vals.index(params[p2])
                j = p1_vals.index(params[p1])
                grid[i, j] = r['avg_acc']

        im = ax.imshow(grid, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(p1_vals)))
        ax.set_xticklabels([f'{v:g}' for v in p1_vals], rotation=45)
        ax.set_yticks(range(len(p2_vals)))
        ax.set_yticklabels([f'{v:g}' for v in p2_vals])
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(f'{p3}={p3_val}')

        for i in range(len(p2_vals)):
            for j in range(len(p1_vals)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f'{grid[i, j]:.1f}', ha='center', va='center',
                             color='white', fontsize=8)

        fig.colorbar(im, ax=ax, label='Average balanced accuracy (%)')

    plt.tight_layout()
    plt.savefig(out, dpi=500)
    plt.show()


def main():
    """Main caller function for CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=100,
                        help='The number of iterations the model should run. We recommend n>10')
    parser.add_argument('--database', type=str,
                        help='Choose the dataset: lexibank, grambank, or combined')
    parser.add_argument("--minimum", action="store", type=int, default=5,
                        help="select the minimum number of languages per family")
    parser.add_argument("--experiment", action="store_true", default=False,
                        help="conduct experiment with full database")
    parser.add_argument('--sweep', action="store_true", default=False,
                     help='Run a hyperparameter sweep instead of a normal training run')
    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    main()
