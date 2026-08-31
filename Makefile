PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
WORKERS ?= 4

.PHONY: help venv setup test smoke experiment-reduced experiment-acrobot \
        experiment-main experiment-ablation experiment-uncertainty \
        experiment-full plots clean

help:
	@echo "make setup                 create .venv and install dependencies"
	@echo "make test                  run the test suite"
	@echo "make smoke                 3k-step end-to-end run on CartPole"
	@echo "make experiment-reduced    CartPole, all variants, 3 seeds (~15-20 min on 4 cores)"
	@echo "make experiment-acrobot    Acrobot, all variants, 3 seeds (~15-20 min on 4 cores)"
	@echo "make experiment-main       LunarLander, all variants, 3 seeds (~3-4 h on 4 cores --"
	@echo "                           LunarLander runs ~8x slower per step than classic control)"
	@echo "make experiment-ablation   Q-scaling ablation on Acrobot (~7 min)"
	@echo "make experiment-uncertainty  uncertainty-gated extension study on Acrobot (~12 min)"
	@echo "make experiment-full       full study: 3 envs x 8 arms x 5 seeds, full step budgets"
	@echo "make plots SWEEP=main_gym  regenerate figures and tables for one sweep"
	@echo ""
	@echo "Override parallelism with WORKERS=N (default 4)."

venv:
	python3 -m venv .venv
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

clean:
	rm -rf results/runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
