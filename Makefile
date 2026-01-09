.PHONY: test run setup

test:
	@. ./scripts/setup.sh && python -m unittest

run:
	@. ./scripts/setup.sh && python enb/app.py
