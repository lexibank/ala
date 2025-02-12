# Automated Language Affiliation (ALA): A Pipeline for Cross-Linguistic Data Formats

With ALA, we offer a service for the automated affiliation of a language, based on standardized data from the Lexibank repository for CLDF wordlists. Language Affiliation is hereby understood as the task by which one tries to identify to which language family a given language belongs.

## Installation Instructions

In order to run all the code, you can install the packages from the `requirements.txt` file (the command `python` here refers to the Python version that you use on your computer, on some systems, this may result in the command `python3`, but we use `python` as a generic placeholder here). You should make sure to use a fresh virtual environment before installing any packages.

```shell
python3 -m pip install -r requirements.txt
python3 -m pip install .
```

## Downloading SQLITE Data for Lexibank, ASJP, and Grambank

In order to download the data necessary for the experiments, you can run the Makefile:

```shell
cd examples/
make download
make prepare
```

## Running the experiments

The experiments can be reproduced by running the following python command:

```shell
python3 -u ala_ff_torch.py --database=lexibank
```

You can chose the database between `lexibank`, `grambank`, `asjp`, or `combined` for the intersection of `lexibank` and `grambank`. Additionally, you can add a flag `-experiment` if you want to test for the linguistic isolates and large language families as specified in the paper.

```shell
python3 -u ala_ff_torch.py --database=grambank -experiment
```
