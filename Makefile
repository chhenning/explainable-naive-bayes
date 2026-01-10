.PHONY: test run setup list_datasets

test:
	@. ./scripts/setup.sh && python -m unittest

run:
	@. ./scripts/setup.sh && python enb/app.py

list_datasets:
	@. ./scripts/setup.sh && clear && python enb/app.py ls