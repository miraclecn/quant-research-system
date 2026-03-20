PYTHON := ./.venv/bin/python
AQT := ./.venv/bin/aqt
INPUT ?= data/daily_bars.parquet
DB ?= stock_data.duckdb
OUTPUT_DIR ?= outputs
START_DATE ?=
END_DATE ?=
RESEARCH_START ?= 2020-01-01
RESEARCH_END ?= 2026-03-20
TRAIN_MONTHS ?= 36
VALID_MONTHS ?= 12
TEST_MONTHS ?= 12
STEP_MONTHS ?= 12
LOOKBACK_DAYS ?= 7

.PHONY: help install run research-run factor-chain single-factor family-sma family-reversal export-panel update-raw update-index-weight update-index-daily daily-research check-daily status snapshot summary

help:
	@printf '%s\n' \
	'Available targets:' \
	'  make install              Install the project into .venv' \
	'  make run                  Run the default pipeline' \
	'  make research-run         Run rolling train/valid/test research' \
	'  make factor-chain         Run factor-chain feature screening' \
	'  make single-factor        Run single-factor diagnostics' \
	'  make family-sma           Run the SMA factor family lab' \
	'  make family-reversal      Run the reversal factor family lab' \
	'  make export-panel         Export panel data from DuckDB' \
	'  make update-raw           Refresh raw Tushare tables' \
	'  make update-index-weight  Refresh index constituent history' \
	'  make update-index-daily   Refresh official index daily bars' \
	'  make daily-research       Run the one-shot daily workflow' \
	'  make check-daily          Check daily output health' \
	'  make status               Show git status and recent commits' \
	'  make snapshot             Show git and artifact snapshot' \
	'  make summary              Generate docs/generated-summary.md'

install:
	$(PYTHON) -m pip install -e .

run:
	$(AQT) run --input $(INPUT) --output-dir $(OUTPUT_DIR)

research-run:
	$(AQT) research-run --input $(INPUT) --output-dir $(OUTPUT_DIR) \
		--research-start $(RESEARCH_START) --research-end $(RESEARCH_END) \
		--train-months $(TRAIN_MONTHS) --valid-months $(VALID_MONTHS) \
		--test-months $(TEST_MONTHS) --step-months $(STEP_MONTHS)

factor-chain:
	$(AQT) factor-chain-run --input $(INPUT) --output-dir $(OUTPUT_DIR) \
		--research-start $(RESEARCH_START) --research-end $(RESEARCH_END) \
		--train-months $(TRAIN_MONTHS) --valid-months $(VALID_MONTHS) \
		--test-months $(TEST_MONTHS) --step-months $(STEP_MONTHS)

single-factor:
	$(AQT) single-factor-run --input $(INPUT) --output-dir $(OUTPUT_DIR) \
		--research-start $(RESEARCH_START) --research-end $(RESEARCH_END)

family-sma:
	$(AQT) family-lab --input $(INPUT) --output-dir $(OUTPUT_DIR) \
		--family sma --research-start $(RESEARCH_START) --research-end $(RESEARCH_END)

family-reversal:
	$(AQT) family-lab --input $(INPUT) --output-dir $(OUTPUT_DIR) \
		--family reversal --research-start $(RESEARCH_START) --research-end $(RESEARCH_END)

export-panel:
	$(AQT) export-panel --input $(DB) --output data/daily_bars.parquet \
		$(if $(START_DATE),--start-date $(START_DATE),) \
		$(if $(END_DATE),--end-date $(END_DATE),)

update-raw:
	$(AQT) update-raw --input $(DB) --lookback-days $(LOOKBACK_DAYS) \
		$(if $(START_DATE),--start-date $(START_DATE),) \
		$(if $(END_DATE),--end-date $(END_DATE),)

update-index-weight:
	$(AQT) update-index-weight --input $(DB) \
		$(if $(START_DATE),--start-date $(START_DATE),) \
		$(if $(END_DATE),--end-date $(END_DATE),)

update-index-daily:
	$(AQT) update-index-daily --input $(DB) \
		$(if $(START_DATE),--start-date $(START_DATE),) \
		$(if $(END_DATE),--end-date $(END_DATE),)

daily-research:
	bash scripts/daily_research.sh

check-daily:
	bash scripts/check_daily_outputs.sh

status:
	git status --short --branch
	git log --oneline --decorate -n 5

snapshot:
	bash scripts/status_snapshot.sh

summary:
	bash scripts/generate_summary.sh
