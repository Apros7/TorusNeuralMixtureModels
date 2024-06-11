MAIN_FOLDER := $(shell git rev-parse --show-toplevel)

install:
	python3 -m pip install -r requirements.txt

syn:
	cd $(MAIN_FOLDER) && python3 src/data/synthetic_data.py