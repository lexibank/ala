"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, concept2vec, feature2vec, get_db, get_asjp
from ala.utils import write_table, main, extract_branch
from ala.model import train


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    wl = load_data('lexibank', args.minimum, args.experiment)
    gb = load_data('grambank', args.minimum, args.experiment)
    asjp = get_asjp()

    gb_conv, _ = feature2vec(get_db('data/grambank.sqlite3'))
    lb_conv, _ = concept2vec(
        get_db('data/lexibank.sqlite3'), model='dolgo', conceptlist="Swadesh-1955-100")

    # Set up combination of LB and GB
    data = convert_data(wl, {k: v[0] for k, v in asjp.items() if k in gb},
                        lb_conv, load='lexical', threshold=args.minimum)
    gb_wl = convert_data(gb, {k: v[0] for k, v in asjp.items() if k in wl},
                         gb_conv, load='grambank', threshold=args.minimum)

    # Combine data vectors
    for lang in data:
        data[lang][2] += gb_wl[lang][2]

    northern_uto = extract_branch(gcode='nort2953')
    anatolian = extract_branch(gcode='anat1257')
    tocharian = extract_branch(gcode='tokh1241')
    sinitic = extract_branch(gcode='sini1245')
    isolates = ['bang1363', 'basq1248', 'mapu1245', 'kusu1250', 'cara1273']

    test_langs = {lang: data[lang] for lang in data if (args.experiment and (
        any(lang in x for x in [sinitic, northern_uto, anatolian, tocharian])
        or lang in isolates))}

    result_per_fam, results_experiment = train(
        data, args.runs, test_langs=test_langs, experiment=args.experiment,
        test_size=0.05)

    # Only write overall results if not runnings the experiment
    if args.experiment is False:
        write_table('combined', result_per_fam, print_table=True)

    # Only write 'experiments_' results in the corresponding setting
    if args.experiment:
        write_table('combined', results_experiment, experiment=args.experiment,  print_table=True)


run_ala(main())
