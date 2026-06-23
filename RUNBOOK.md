# Runbook

Exact command sequence from a clean checkout to a finished report.

## Phase 0 -- local machine (no GPU needed)

```bash
cd /goinfre/injo/gpu
python -m venv .venv-local && source .venv-local/bin/activate
pip install -r requirements-local.txt

python data/prep_scripts/prepare_task_a_bfcl.py
python data/prep_scripts/prepare_task_b_livecodebench.py
python data/prep_scripts/prepare_task_c_niah.py

pytest tests/ -v
```

Request gated access to `meta-llama/Llama-3.3-70B-Instruct` on Hugging Face
**now** -- approval can take hours to a few days and should run in parallel
with everything else, not block it.

```bash
git init
git add .
git commit -m "Initial scaffold"
# rsync or git clone this repo to the rented H100 box, e.g.:
rsync -avz --exclude .venv-local --exclude data/raw --exclude results . user@h100box:~/gpu/
```

## Phase 1 -- remote H100 box, one-time setup

```bash
ssh user@h100box
cd gpu/
cp .env.example .env   # fill in HF_TOKEN
export $(grep -v '^#' .env | xargs)   # or use direnv/dotenv loader of choice

bash setup/install.sh
source venv/bin/activate
python setup/verify_env.py   # confirms H100 80GB detected, vllm/bnb importable, writes results/env_info.json

python setup/download_models.py --model qwen2.5-32b
python setup/download_models.py --model gemma4-31b
python setup/download_models.py --model mixtral-8x7b
python setup/download_models.py --model llama3.3-70b   # only once gated access is approved
```

Confirm at least 500GB-1TB free disk (`verify_env.py` warns if not) -- 4
models at native precision total ~360GB, and the bnb config-patch fallback
(if triggered) can roughly double a given model's on-disk footprint.

## Phase 2 -- spike tests (CRITICAL, do not skip)

```bash
python spike_tests/spike_test_bnb_quant_args.py
python spike_tests/spike_test_mixtral_bnb.py
python spike_tests/spike_test_gemma4_bnb.py
```

If `spike_test_bnb_quant_args.py` FAILs: re-run the full matrix in Phase 3
with `--quant-config-mode patched_config`.

If `spike_test_mixtral_bnb.py` or `spike_test_gemma4_bnb.py` FAILs: see the
fallback notes printed by the script and in `configs/run_matrix.yaml`'s
`known_risk` field before proceeding -- do not run those legs of the matrix
until resolved or explicitly accepted as a documented limitation.

## Phase 3 -- the main run (sequential, 14 combos x 3 tasks)

```bash
python -m benchmark.orchestrator --config configs/run_matrix.yaml
# resume after an interruption:
python -m benchmark.orchestrator --resume
# re-run just one model after a config fix:
python -m benchmark.orchestrator --only mixtral-8x7b --resume
```

Rough budget: model download 1.5-3h, ~25-40 min/combo x 14 ≈ 7-10h pure
compute, 12-18h with retries/larger-model overhead. Plan 2-3 calendar days
total including the Llama gated-access wait. Recommend running unattended
overnight.

## Phase 4 -- aggregate + report

```bash
python aggregate/aggregate_results.py
python report/generate_report.py
# inspect report/report.md and report/figures/*.png
```

## Phase 5 -- pull results back to the local machine

```bash
# from the local machine
rsync -avz user@h100box:~/gpu/results/ ./results/
rsync -avz user@h100box:~/gpu/report/ ./report/
```
