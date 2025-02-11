import argparse
import csv
import os
from statistics import mean, stdev
from clldutils.clilib import add_format, Table


def write_table(database, results, intersection, print_table):
    """
    Writes the results to a file and optionally also prints a table to the command line.
    """
    tag = 'results_' if intersection else 'experiments_'
    output = os.path.join('results', tag + database + '.tsv')
    with open(output, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Run', 'Family', 'Languages', 'Tested', 'Score'])
        for _, rows in results.items():
            writer.writerows(rows)

    if print_table:
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
    parser.add_argument("--intersection", action="store", type=bool, default=True,
                        help="only add languages common to all databases")
    add_format(parser, default='simple')
    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    main()
