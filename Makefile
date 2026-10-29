# HashedContext reproducibility package. Run `make help` for the list of targets.
PY ?= .venv/bin/python

.PHONY: help setup test quick full live compare tables clean-results

help:
	@echo "make setup    create .venv, install pinned dependencies, fetch the 4 pinned repos"
	@echo "make test     run the unit and property tests (~10 s)"
	@echo "make quick    small-scale run of every deterministic part + comparison (~5 min)"
	@echo "make full     reproduce every published number except the live-LLM test (20 min - 2 h)"
	@echo "make live     live-LLM diff-chain test (needs HC_API_BASE, HC_MODEL and a key file)"
	@echo "make compare  compare results/ with published-results/"
	@echo "make tables   rebuild the trade-off tables from results/"
	@echo "make clean-results  delete your own results/ and figures/ (published-results/ is kept)"

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip -q
	.venv/bin/pip install -r requirements.txt -q
	.venv/bin/python -m sim.fetch_corpus

test:
	$(PY) -m pytest -q tests

quick:
	scripts/quick_check.sh

full:
	scripts/reproduce_all.sh

live:
	scripts/run_live_llm.sh

compare:
	$(PY) scripts/compare_with_published.py

tables:
	$(PY) -m experiments.tradeoff_tables

clean-results:
	rm -rf results figures
