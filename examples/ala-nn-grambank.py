"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, feature2vec, get_db, get_asjp
from ala.utils import write_table, main, extract_branch
from ala.model import train


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    gb = load_data('grambank', args.minimum, args.experiment)
    asjp = get_asjp()

    gb_conv, _ = feature2vec(get_db('data/grambank.sqlite3'))
    data = convert_data(gb, {k: v[0] for k, v in asjp.items()},
                        gb_conv, load='grambank', threshold=args.minimum)

    northern_uto = extract_branch(gcode='nort2953')
    anatolian = extract_branch(gcode='anat1257')
    tocharian = extract_branch(gcode='tokh1241')
    sinitic = extract_branch(gcode='sini1245')
    isolates = ['bang1363', 'basq1248', 'mapu1245', 'kusu1250']

    test_langs = {lang: data[lang] for lang in data if (args.experiment and (
        any(lang in x for x in [sinitic, northern_uto, anatolian, tocharian])
        or lang in isolates))}

    result_per_fam, results_experiment = train(
        data, args.runs, test_langs=test_langs, experiment=args.experiment)

    # Only write overall results if not runnings the experiment
    if args.experiment is False:
        write_table('grambank', result_per_fam, experiment=args.experiment, print_table=True)

    # Only write 'experiments_' results in the corresponding setting
    if args.experiment:
        write_table('grambank', results_experiment, experiment=args.experiment,  print_table=True)


run_ala(main())
