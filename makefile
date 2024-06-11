
install:
	python3 -m pip install -r requirements.txt

test:
	@echo 'testing...'

syn:
	python3 src/data/synthetic_data.py