import argparse
import csv
import os
from collections import Counter
from statistics import mean, stdev
from clldutils.clilib import add_format, Table
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
    Retrieves all glottocodes that are part of a certain branch
    in Glottolog.
    """
    gcode = "%" + gcode + "%"
    db = get_db("data/glottolog.sqlite3")
    db.execute(BRANCH_QUERY, (gcode,))
    gcodes = [glottocode[0] for glottocode in db.fetchall()]

    return gcodes


def write_table(database, results, experiment, print_table):
    """
    Writes the results to a file and optionally also prints a table to the command line.
    """
    if experiment is True:
        results_table = []
        for item in results:
            for k, v in Counter(results[item]).items():
                results_table.append([item[0], item[1], k, v])
        output = 'results/experiment_' + database + '.tsv'
        header = ['Language', 'Family', 'Prediction', 'Frequency']

        output = os.path.join('results', 'experiments_' + database + '.tsv')
        with open(output, 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(header)
            writer.writerows(results_table)

    else:
        output = os.path.join('results', 'results_' + database + '.tsv')
        with open(output, 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Run', 'Family', 'Languages', 'Tested', 'Score'])
            for fam, rows in results.items():
                if fam != 'TOTAL':
                    writer.writerows(rows)

    if print_table and experiment is False:
        # Summary table for command line
        table = [[
            fam, mean([r[2] for r in rows]),
            round(mean([r[3] for r in rows]), 1),   # Tested langs
            round(mean([r[4] for r in rows]), 2),   # Acc
            0 if len(rows) < 2 else round(stdev([r[4] for r in rows]), 2)]
            for fam, rows in sorted(results.items())
        ]

        total = [fam for fam in table if fam[0] == 'TOTAL']
        new_table = [fam for fam in table if fam[0] != 'TOTAL'] + total

        with Table(*['Family', 'Languages', 'Tested', 'Avg. Fam. Accuracy', 'Fam-STD']) as t:
            t.extend(new_table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=100,
                        help='The number of iterations the model should run. We recommend n>10')
    parser.add_argument('--database', type=str,
                        help='Choose the dataset: lexibank, grambank, or combined')
    parser.add_argument("--minimum", action="store", type=int, default=5,
                        help="select the minimum number of languages per family")
    parser.add_argument("--experiment", action="store_true", default=False,
                        help="conduct experiment with full database")
    add_format(parser, default='simple')
    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    main()
