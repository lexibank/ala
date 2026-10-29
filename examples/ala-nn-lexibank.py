"""
Module for running the experiments for Automated Language Affiliation.
"""
from ala.data import convert_data, load_data, concept2featurevec, get_db
from ala.utils import write_table, main
from ala.model import run_model, TrainConfig


def run_ala(args):
    """Defines the workflow for data loading in the different settings."""
    # get configuration options
    conceptlist='Starostin-1991-110'
    # conceptlist="Swadesh-1955-100"
    # conceptlist="Tadmor-2009-100"
    # conceptlist="Swadesh-1952-200"

    wl, families = load_data('lexibank', args.minimum, args.experiment, conceptlist)

    lb_conv = concept2featurevec(get_db('data/lexibank.sqlite3'), conceptlist)

    data = convert_data(wl, families, lb_conv, load='lexical')

    isolates = [
    "mose1249", "movi1243", "lule1238", "cams1241", "puin1248", "paez1247",
    "ando1256", "cand1248", "cofa1242", "yuwa1244", "waor1240", "maku1246",
    "pume1238", "taus1253", "urar1246", "wara1303", "kwaz1243", "trum1247",
    "kunz1244", "puel1244", "yama1264", "anda1286", "cayu1262", "iton1250",
    "kano1245", "muni1258", "pira1253", "yura1255", "cani1243", "mapu1245",
    "cara1273", "puqu1242",
    # Mochica
    "moch1259",
    # Peba-Yagua
    'yagu1244', 'peba1243', 'yame1242',
    # Zaparoan
    'ando1255', 'zapa1253', 'iqui1243', 'arab1268',
    # Chicham
    #'achu1248', "huam1247", "agua1253", "shua1257",
    # Cahuapanan
    "jebe1250", "chay1248"
    # check if Arawan and Harakmbut are added correctly as languages
    ]

    test_langs = {lang: data[lang] for lang in data if (args.experiment and lang in isolates)}

    # For concept2vec
    #config = TrainConfig(lr=1e-3)

    # # For concept2featurevec
    config = TrainConfig(lr=5e-4)

    result_per_fam, results_experiment = run_model(data, args.runs, config, test_langs=test_langs)

    # Only write overall results if not runnings the experiment
    if args.experiment is False:
        write_table('lexibank', result_per_fam)

    # Only write 'experiments_' results in the corresponding setting
    if args.experiment:
        write_table('lexibank', results_experiment, experiment=True, print_table=False)


run_ala(main())
