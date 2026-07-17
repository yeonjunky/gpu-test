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
source .venv/bin/activate
python setup/verify_env.py   # confirms H100 80GB detected, vllm/bnb importable, writes results/env_info.json

python setup/download_models.py --model qwen2.5-32b
python setup/download_models.py --model gemma4-31b
python setup/download_models.py --model mixtral-8x7b
python setup/download_models.py --model llama3.3-70b   # only once gated access is approved
```

Confirm at least 500GB-1TB free disk (`verify_env.py` warns if not) -- 4
models at native precision total ~360GB, plus a locally-quantized cache per
bnb `quant_level` (`/tmp/quantized_models/` by default, see Phase 2 note
below), which can roughly double the on-disk footprint across the matrix.

## Phase 2 -- spike tests (CRITICAL, do not skip)

```bash
python spike_tests/spike_test_mixtral_bnb.py
python spike_tests/spike_test_gemma4_bnb.py
```

Note: `spike_test_bnb_quant_args.py` no longer needs to be run. Its original
purpose was to check whether vLLM's `hf_overrides={"quantization_config":
{...}}` correctly propagates `bnb_4bit_use_double_quant` -- that mechanism
was confirmed broken (a weight-shape `AssertionError` in vLLM's loader,
reproduced across every vLLM version tested, 0.9.2 through 0.25.1; see
`spike_test_error_report.md` for the full investigation). `benchmark/
engine.py::build_llm` no longer uses `hf_overrides` or a bare `quantization=`
argument for bnb rungs at all -- it pre-quantizes each (model, quant_level)
offline with `transformers` + `BitsAndBytesConfig`, saves the real quantized
checkpoint to a local cache dir, and points vLLM at that cache dir directly
(the officially documented "pre-quantized checkpoint" loading pattern,
rather than in-flight quantization via a config override). The
`--quant-config-mode` flag on `run_one_combo.py`/`orchestrator.py` and the
`patched_config` fallback it used to select have been removed -- both were
broken the same way.

`spike_test_mixtral_bnb.py` and `spike_test_gemma4_bnb.py` are still
relevant -- MoE+bnb instability and gemma4's untested-with-bnb status are
real, architecture-specific risks independent of the fix above -- but as
currently written they still exercise the old `hf_overrides` path, not the
new `build_llm()` path, so their PASS/FAIL isn't representative of what
Phase 3 will actually do. Until they're adapted to call
`benchmark.engine.build_llm` directly, spot-check the real Mixtral/gemma4
bnb legs against `build_llm` before trusting Phase 3 for those two models.

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
