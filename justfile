default: test-all

test-all:
	pytest tests

test-models:
	pytest tests/models