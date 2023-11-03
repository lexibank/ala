download:
	git clone https://github.com/glottolog/glottolog-cldf data/glottolog
	git clone https://github.com/lexibank/lexibank-analysed.git data/lexibank-analysed
	git clone https://github.com/lexibank/asjp.git data/asjp
	git clone https://github.com/grambank/grambank.git data/grambank
	git clone https://github.com/pano-tacanan-history/blumpanotacana data/blumpanotacana
	git clone https://github.com/lexibank/iecor data/iecor
	git clone https://github.com/lexibank/grollemundbantu data/grollemundbantu
	git clone https://github.com/lexibank/birchallchapacuran data/birchallchapacuran
	git clone https://github.com/lexibank/viegasbarroschaco	data/viegasbarroschaco

prepare:
	pip install pycldf, scipy, torch
	cd data/glottolog; git checkout v4.8; cd ../../
	cd data/lexibank-analysed; git checkout bliss2; cd ../../
	cd data/asjp; git checkout v20; cd ../../
	cd data/grambank; git checkout v1.0.3; cd ../../
	cd data/iecor; git checkout v1.0; cd ../../
	cd data/grollemundbantu; git checkout v1.0; cd ../../
	cd data/birchallchapacuran; git checkout v1.1.0; cd ../../
	cldf createdb data/glottolog/cldf/cldf-metadata.json data/glottolog.sqlite3
	cldf createdb data/lexibank-analysed/cldf/wordlist-metadata.json data/lexibank.sqlite3
	cldf createdb data/asjp/cldf/cldf-metadata.json data/asjp.sqlite3
	cldf createdb data/grambank/cldf/StructureDataset-metadata.json data/grambank.sqlite3
	cldf createdb data/blumpanotacana/cldf/cldf-metadata.json data/blumpanotacana.sqlite3
	cldf createdb data/birchallchapacuran/cldf/cldf-metadata.json data/birchallchapacuran.sqlite3
	cldf createdb data/iecor/cldf/cldf-metadata.json data/iecor.sqlite3
	cldf createdb data/grollemundbantu/cldf/cldf-metadata.json data/grollemundbantu.sqlite3
	cldf createdb data/viegasbarroschaco/cldf/cldf-metadata.json data/viegasbarroschaco.sqlite3
