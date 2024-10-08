download:
	git clone https://github.com/glottolog/glottolog-cldf data/glottolog
	git clone https://github.com/lexibank/lexibank-analysed.git data/lexibank-analysed
	git clone https://github.com/lexibank/asjp.git data/asjp
	git clone https://github.com/grambank/grambank.git data/grambank

prepare:
  pip install pycldf scipy torch
	cd data/glottolog; git checkout v5.0; cd ../../	
	cd data/lexibank-analysed; git checkout bliss2; cd ../../
	cd data/asjp; git checkout v20; cd ../../
	cd data/grambank; git checkout v1.0.3; cd ../../
	cldf createdb data/glottolog/cldf/cldf-metadata.json data/glottolog.sqlite3
	cldf createdb data/lexibank-analysed/cldf/wordlist-metadata.json data/lexibank.sqlite3
	cldf createdb data/asjp/cldf/cldf-metadata.json data/asjp.sqlite3
	cldf createdb data/grambank/cldf/StructureDataset-metadata.json data/grambank.sqlite3
