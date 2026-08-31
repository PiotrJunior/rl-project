PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
WORKERS ?= 4

# Interpreter used to CREATE the venv. Needs >= 3.10: config loading resolves
# `X | None` annotations at runtime via get_type_hints. macOS ships 3.9 as
# `python3`, so override this if `make setup` fails on the venv step:
#   make setup PYTHON=python3.12
PYTHON ?= python3

.PHONY: help venv setup test smoke experiment-reduced experiment-acrobot \
        experiment-main experiment-ablation experiment-uncertainty \
        experiment-full plots paper clean

help:
	@echo "make setup                 create .venv and install dependencies"
	@echo "make test                  run the test suite"
	@echo "make smoke                 3k-step end-to-end run on CartPole"
	@echo "make experiment-reduced    CartPole, all variants, 3 seeds (~2 min, WORKERS=8)"
	@echo "make experiment-acrobot    Acrobot, all variants, 3 seeds (~3 min, WORKERS=8)"
	@echo "make experiment-main       LunarLander, all variants, 3 seeds, full 400k budget (~12 min, WORKERS=8)"
	@echo "make experiment-ablation   Q-scaling ablation on Acrobot (~2 min)"
	@echo "make experiment-uncertainty  uncertainty-gated extension study on Acrobot (~3 min)"
	@echo "make experiment-full       THE study: 3 envs x 8 arms x 5 seeds (~35 min, WORKERS=8)"
	@echo "make plots SWEEP=full_gym  regenerate figures and tables for one sweep"
	@echo "make paper                 build paper/paper.tex -> paper/paper.pdf"
	@echo ""
	@echo "Override parallelism with WORKERS=N (default 4; timings above assume 8)."

venv:
	$(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip

# swig MUST be installed before gymnasium[box2d]: box2d-py has no prebuilt
# wheels for CPython 3.11+ on linux-x86_64 or macOS-arm64 and compiles from
# source, which needs swig on PATH. Installing them together fails.
setup: venv
	$(PIP) install swig
	$(PIP) install -r requirements.txt
	$(PY) -c "import gymnasium; gymnasium.make('LunarLander-v3'); print('Box2D OK')"

test:
	PYTHONPATH=src $(PY) -m pytest tests/ -q

smoke:
	PYTHONPATH=src $(PY) -m e2b.train --config experiments/smoke

experiment-reduced:
	PYTHONPATH=src $(PY) scripts/run_sweep.py --sweep reduced_gym --workers $(WORKERS)
	$(MAKE) plots SWEEP=reduced_gym

experiment-acrobot:
	PYTHONPATH=src $(PY) scripts/run_sweep.py --sweep acrobot_gym --workers $(WORKERS)
	$(MAKE) plots SWEEP=acrobot_gym

experiment-main:
	PYTHONPATH=src $(PY) scripts/run_sweep.py --sweep main_gym --workers $(WORKERS)
	$(MAKE) plots SWEEP=main_gym

experiment-ablation:
	PYTHONPATH=src $(PY) scripts/run_sweep.py --sweep ablation_scaling --workers $(WORKERS)
	$(MAKE) plots SWEEP=ablation_scaling

experiment-uncertainty:
	PYTHONPATH=src $(PY) scripts/run_sweep.py --sweep uncertainty --workers $(WORKERS)
	$(MAKE) plots SWEEP=uncertainty

experiment-full:
	PYTHONPATH=src $(PY) scripts/run_sweep.py --sweep full_gym --workers $(WORKERS)
	$(MAKE) plots SWEEP=full_gym

SWEEP ?= reduced_gym
plots:
	PYTHONPATH=src $(PY) scripts/make_plots.py --sweep $(SWEEP)

# Needs a TeX distribution (MacTeX / TeX Live). Figures come from
# report/figures/, so run `make plots` first if they are stale.
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode paper.tex

clean:
	rm -rf results/runs .pytest_cache
	cd paper && latexmk -C 2>/dev/null || true
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
