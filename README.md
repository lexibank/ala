# Automated Language Affiliation (ALA): A Pipeline for Cross-Linguistic Data Formats

With ALA, we offer a service for the automated affiliation of a language, based on standardized data from the Lexibank repository for CLDF wordlists. Language Affiliation is hereby understood as the task by which one tries to identify to which language family a given language belongs.

## Citation

If you use our model or workflow, please cite our paper as follows:

> Blum, Frederic and Herbold, Steffen and List, Johann-Mattis (forthcoming): From Isolates to Families: Using Neural Networks for Automated Language Affiliation. In: Proceedings of the Association of Computational Linguistics 2025. 1-12. https://doi.org/10.48550/arXiv.2502.11688

## Installation Instructions

In order to run all the code, you can install the packages from the `requirements.txt` file (the command `python` here refers to the Python version that you use on your computer, on some systems, this may result in the command `python3`, but we use `python` as a generic placeholder here). You should make sure to use a fresh virtual environment before installing any packages.

```shell
python -m pip install -r requirements.txt
python -m pip install .
```

## Downloading SQLITE Data for Lexibank, ASJP, and Grambank

In order to download the data necessary for the experiments, you can run the Makefile, by opening a terminal in the `examples` folder and installing the additional packages:

```shell
cd examples
python -m pip install -r requirements.txt
```

You can now download the data and create the SQLite databases.

```shell
make download
make prepare
```

## Running the Experiments

To run the experiments, you should open a terminal in the `examples` folder. Here, we have prepared shell scripts that call the Python commands for convenience.

```shell
sh runs-bl.sh
sh runs-comparison-nn.sh
sh runs-experiments.sh
```
