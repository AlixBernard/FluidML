default: test-all

init:
	pip install -r requirements.txt

init-dev:
	pip install -r requirements.txt
	pip install -r dev-requirements.txt

test-all:
	pytest --cov=fluidml tests

test-models:
	pytest --cov=fluidml/models tests/models
