default: test-all

install-requirements-all:
	pip install -r requirements.txt
	pip install -r dev-requirements.txt

test-all:
	pytest tests

test-models:
	pytest tests/models