"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, concept2vec, get_db, get_asjp, get_other
from ala.utils import write_table, main, extract_branch
from ala.model import train


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    wl = load_data('lexibank', args.minimum, args.experiment)
    asjp = get_asjp()

    lb_conv, _ = concept2vec(get_db('data/lexibank.sqlite3'), model='dolgo',
                             conceptlist="Swadesh-1955-100")

    data = convert_data(wl, {k: v[0] for k, v in asjp.items()},
                        lb_conv, load='lexical', threshold=args.minimum)

    if args.experiment:
        data['cara1273'] = convert_data(dict(get_other(mode="carari").items()),
                                        {'cara1273': "Unclassified"},
                                        lb_conv, threshold=1)['cara1273']

    northern_uto = extract_branch(gcode='nort2953')
    anatolian = extract_branch(gcode='anat1257')
    tocharian = extract_branch(gcode='tokh1241')
    sinitic = extract_branch(gcode='sini1245')
    isolates = ['bang1363', 'basq1248', 'mapu1245', 'kusu1250', 'cara1273']

    test_langs = {lang: data[lang] for lang in data if (args.experiment and (
        any(lang in x for x in [sinitic, northern_uto, anatolian, tocharian])
        or lang in isolates))}

    result_per_fam, results_experiment = train(
        data, args.runs, test_langs=test_langs, experiment=args.experiment)

    # Only write overall results if not runnings the experiment
    if args.experiment is False:
        write_table('lexibank', result_per_fam, experiment=args.experiment, print_table=True)

    # Only write 'experiments_' results in the corresponding setting
    if args.experiment:
        write_table('lexibank', results_experiment, experiment=args.experiment,  print_table=True)


run_ala(main())
