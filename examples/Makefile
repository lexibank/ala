download:
	git clone https://github.com/lexibank/lexibank-analysed.git
	git clone https://github.com/lexibank/asjp.git
	git clone https://github.com/grambank/grambank.git
	git clone https://github.com/pano-tacanan-history/blumpanotacana

prepare:
	pip install pycldf
	cd lexibank-analysed; git checkout bliss2
	cd asjp; git checkout v20
	cd grambank; git checkout v1.0.3
	cldf createdb lexibank-analysed/cldf/wordlist-metadata.json lexibank.sqlite3
	cldf createdb asjp/cldf/cldf-metadata.json asjp.sqlite3
	cldf createdb grambank/cldf/StructureDataset-metadata.json grambank.sqlite3
	cldf createdb blumpanotacana/cldf/cldf-metadata.json blumpanotacana.sqlite3
