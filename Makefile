.PHONY: test help run ls

test:
	@. ./scripts/setup.sh && python -m unittest

help:
	@. ./scripts/setup.sh && clear && python enb/app.py --help

run:
	@. ./scripts/setup.sh && python enb/app.py run

ls:
	@. ./scripts/setup.sh && clear && python enb/app.py ls