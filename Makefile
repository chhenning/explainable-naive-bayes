
test:
	@. ./scripts/setup.sh && python -m unittest

help:
	@. ./scripts/setup.sh && clear && python enb/app.py --help

run:
	@. ./scripts/setup.sh && clear && python enb/app.py run --cm

explain:
	@. ./scripts/setup.sh && clear && python enb/app.py run -ds "fake_newsgroup" --explain "Public opinion on firearm regulation varies widely by region, often influenced by cultural and historical factors."

ls:
	@. ./scripts/setup.sh && clear && python enb/app.py ls

.PHONY: test help run explain ls
