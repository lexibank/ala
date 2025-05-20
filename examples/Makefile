download:
	git clone https://github.com/glottolog/glottolog-cldf data/glottolog --depth 1 --branch v5.1
	git clone https://github.com/lexibank/lexibank-analysed.git data/lexibank-analysed --depth 1 --branch v2.1
	git clone https://github.com/lexibank/asjp.git data/asjp --depth 1 --branch v20
	git clone https://github.com/grambank/grambank.git data/grambank --depth 1 --branch v1.0.3

prepare:
	rm data/*.sqlite3
	cldf createdb data/glottolog/cldf/cldf-metadata.json data/glottolog.sqlite3
	cldf createdb data/lexibank-analysed/cldf/wordlist-metadata.json data/lexibank.sqlite3
	cldf createdb data/asjp/cldf/cldf-metadata.json data/asjp.sqlite3
	cldf createdb data/grambank/cldf/StructureDataset-metadata.json data/grambank.sqlite3
