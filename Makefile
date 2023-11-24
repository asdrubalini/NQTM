prepare:
	python utils/preprocess.py --data_path data/stackoverflow --output_dir input/stackoverflow

run:
	python run_NQTM.py --data_dir input/stackoverflow --output_dir output --epoch=2000

evaluate:
	python utils/TU.py --data_path output/top_words_T15_K50_1th

develop:
	NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
