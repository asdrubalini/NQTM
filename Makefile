prepare:
	python3 utils/preprocess.py --data_path data/stackoverflow --output_dir input/stackoverflow

run:
	python3 run_NQTM.py --data_dir input/stackoverflow --output_dir output --epoch=50

evaluate:
	python3 utils/TU.py --data_path output/top_words_T15_K50_1th

develop:
	nix develop ".#devShells.x86_64-linux.default"

store:
	nix-store -qR $(nix path-info ".#devShells.x86_64-linux.default")
