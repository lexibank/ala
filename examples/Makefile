download:
	git clone https://github.com/lexibank/lexibank-analysed.git
	git clone https://github.com/lexibank/asjp.git
        git clone https://github.com/grambank/grambank.git


prepare:
	pip install cldf
	cd lexibank-analysed; git checkout bliss2
	cd asjp; git checkout v20
        cd grambank; git checkout v1.0.3
	cldf createdb lexibank-analysed/cldf/wordlist.json lexibank.sqlite3
	cldf createdb asjp/cldf/cldf-metadata.json asjp.sqlite3
        cldf createdb grambank/cldf/StructureDataset-metadata.json


