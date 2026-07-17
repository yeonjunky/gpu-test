# Spike Test Error Report

**Date:** 2026-07-17
**Environment:** remote H100 NVL box, 95GB VRAM, models already downloaded locally (~340GB in `huggingface/`)
**vLLM version tested:** 0.19.1 (installed in `/root/project/.venv`; `dataset/gpu/.venv` did not exist — Phase 1 of RUNBOOK.md had not been run for this project directory)
**bitsandbytes:** 0.49.2, **transformers:** 5.13.0, **torch:** 2.10.0+cu128

## Summary

All three Phase 2 spike tests fail. The failures are **not** the MoE/Gemma4-specific
risks the scripts' docstrings anticipate — they are a more fundamental break in how
vLLM 0.19.1 handles an explicitly-injected `quantization_config`, whether passed via
`hf_overrides` (the primary path in `benchmark/engine.py::build_llm`) or via a patched
local `config.json` (the fallback path, `build_llm_with_patched_config`, selected with
`--quant-config-mode patched_config`). **Both paths are currently broken**, so the
documented fallback does not actually unblock the matrix as-is.

No lockfile or pinned vLLM version exists anywhere in the repo — `requirements-remote.txt`
only specifies `vllm>=0.9.2` — so it's unknown what version this pipeline was authored/
last verified against. 0.19.1 is simply what was installed in the available venv.

## Test 1: `spike_test_bnb_quant_args.py`

**Result: FAIL** — worker crashes for `double_quant=false` before any VRAM comparison happens.

```
AssertionError
  File ".../vllm/model_executor/layers/linear.py", line 1500, in weight_loader
    assert param_data.shape == loaded_weight.shape
```

Root cause isolated by direct comparison:

- `quantization="bitsandbytes"` **with** `hf_overrides={"quantization_config": {...nf4 args...}}` → crashes with the shape-mismatch `AssertionError` above during weight loading.
- `quantization="bitsandbytes"` + `load_format="bitsandbytes"`, **no** `hf_overrides` (vLLM's own default NF4 quantization) → loads and generates correctly.

So the explicit `quantization_config` injection itself is what breaks weight loading in
this vLLM version — not something specific to the double-quant flag.

Additionally isolated the 8-bit variant (`load_in_8bit: true`, with `quant_method` key
added so it passes config validation): loads without error, but the model **crashes on
the first generate() call**:

```
AttributeError: 'Int8Params' object has no attribute 'bnb_shard_offsets'
  File ".../vllm/model_executor/layers/quantization/bitsandbytes.py", line 303, in _apply_8bit_weight
```

So both the 4-bit and 8-bit explicit-config paths are broken, at different stages
(load time vs. first forward pass).

## Test 2: `spike_test_mixtral_bnb.py`

**Result: FAIL** — crashes before model load even starts, at config validation:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Quantization method specified in the model config (None) does not
  match the quantization method specified in the `quantization` argument (bitsandbytes).
```

Cause: the script's `hf_overrides` block —
```python
hf_overrides={"quantization_config": {"load_in_8bit": True, "load_in_4bit": False}}
```
— is missing a `"quant_method": "bitsandbytes"` key. vLLM 0.19.1 requires this key to
be present and to match the `quantization=` argument; without it, `ModelConfig`
validation rejects the config before any MoE/bnb-specific code runs at all. This means
the test never actually exercises the historical Mixtral+bnb instability it was written
to check for (vllm-project/vllm PR #20061 / issue #20480 / RFC #39583) — it fails for an
unrelated, more basic reason first.

## Test 3: `spike_test_gemma4_bnb.py`

**Result: FAIL** — identical `ValidationError` to Test 2, same root cause (missing
`quant_method` key in `hf_overrides`'s `quantization_config`). Never reaches the
gemma4-specific W4A16/int8-per-channel question the script is meant to probe.

## Fallback path check: `build_llm_with_patched_config` / `--quant-config-mode patched_config`

Manually reproduced this path (writes `bnb_args` into a local copy of `config.json`,
points `LLM(model=...)` at that directory instead of using `hf_overrides`) against the
small test model. **It fails with the exact same `ValidationError` as Tests 2/3**,
because `configs/run_matrix.yaml`'s `bnb_args` blocks also never include a
`quant_method` key. The documented "if hf_overrides fails, switch to patched_config"
escape hatch in RUNBOOK.md Phase 2 does not currently work either — the same missing
key affects both mechanisms.

## Net effect on `configs/run_matrix.yaml`

None of the four bnb rungs used across the matrix (`int8_bnb`, `int4_nf4_bnb`,
`int4_nf4_doublequant_bnb`, and the Mixtral/Llama `int8_baseline`) can currently be
driven through either `quant-config-mode`. Only vLLM's own default NF4 config (reached
by omitting `quantization_config` entirely) loads and generates successfully — which
does not let the run matrix distinguish between int8 vs. int4 vs. double-quant, the
entire point of those rows.

## Version-range check: is this specific to vLLM 0.19.1?

Tried `vllm==0.9.2` (the floor pinned in `requirements-remote.txt`) in an isolated venv
(`dataset/gpu/.venv-vllm092`) to see whether the bugs above are a 0.19.1 regression.

Note: `vllm==0.9.2` requires `transformers>=4.51.1` with no upper bound, so plain
`pip install` initially resolved `transformers==5.14.1` (and separately `4.57.6`), both
of which fail to even import vLLM:

```
ValueError: 'aimv2' is already used by a Transformers config, pick another name.
```

vLLM 0.9.2's `transformers_utils/configs/ovis.py` registers `aimv2` as a custom config
type; recent `transformers` releases added `aimv2` natively, so the two collide.
Pinning down to `transformers==4.51.1` (vLLM 0.9.2's exact declared floor) fixed the
import. With that pin, re-ran `spike_test_bnb_quant_args.py`:

**Result: same failure.** Identical `AssertionError: assert param_data.shape ==
loaded_weight.shape` in `linear.py`'s `weight_loader`, at the same point in weight
loading. So the bug is **not specific to 0.19.1** — it reproduces at the oldest version
this repo claims to support too, once a compatible `transformers` is used. This means
downgrading vLLM alone is unlikely to fix it; the `hf_overrides`-based
`quantization_config` injection approach in `benchmark/engine.py::build_llm` appears to
have never worked correctly across this version range, at least not for the way this
repo constructs the override dict.

## Latest vLLM (0.25.1)

Tried the newest released vLLM (0.25.1, torch 2.11.0+cu130, transformers 5.14.1,
installed cleanly with no import conflicts) in `dataset/gpu/.venv-vllm-latest`.

**Result: same failure.** Identical `AssertionError: assert param_data.shape ==
loaded_weight.shape` in `linear.py`'s `weight_loader`, same code path.

Also tried adding `load_format="bitsandbytes"` explicitly (on the theory that without
it, vLLM might be trying to load the on-disk full-precision weights directly against
quantized parameter shapes declared via `hf_overrides`, without an actual quantize-on-
load step). This did **not** change the outcome — identical assertion, same line.

## Conclusion

The shape-mismatch `AssertionError` reproduces **identically across the entire tested
vLLM range**: 0.9.2 (floor pinned in `requirements-remote.txt`), 0.19.1 (originally
installed), and 0.25.1 (latest release). This rules out a vLLM-version regression as
the cause. The `hf_overrides={"quantization_config": {...}}` mechanism this repo's
`benchmark/engine.py::build_llm` and all three spike test scripts rely on to drive
per-rung bnb settings (int8 vs. int4/nf4, double-quant on/off) does not appear to work
at all for on-the-fly bnb quantization of a full-precision checkpoint, on any vLLM
version tried, for this simple dense Qwen2 architecture — let alone Mixtral (MoE) or
Gemma4.

## Web research: is this a known issue, and what's the correct pattern?

- vLLM's own bnb docs ([docs.vllm.ai/.../quantization/bnb](https://docs.vllm.ai/en/stable/features/quantization/bnb/))
  describe exactly two supported usage patterns, and neither matches what this repo does:
  - **Pre-quantized checkpoint**: point `LLM(model=...)` at a repo/directory whose
    `config.json` *already* contains a real `quantization_config` (e.g. an `unsloth/*-bnb-4bit`
    checkpoint saved by `transformers`). No `quantization=` argument or `hf_overrides` needed —
    vLLM infers everything from the checkpoint itself.
  - **In-flight quantization**: pass `quantization="bitsandbytes"` (and per the docs example,
    also `load_format="bitsandbytes"`) against a full-precision checkpoint that has *no*
    `quantization_config` in its config at all — vLLM applies its own default NF4 settings
    during load. There is no documented option to customize `bnb_4bit_use_double_quant` /
    `bnb_4bit_quant_type` / `bnb_4bit_compute_dtype` for this path.
  - **Neither pattern is "in-flight quantization with a custom `quantization_config` supplied
    via `hf_overrides`"** — which is exactly what `benchmark/engine.py::build_llm` and all
    three spike test scripts do. This isn't a documented/supported vLLM usage at all.
- The exact `AssertionError: assert param_data.shape == loaded_weight.shape` is a recurring,
  **long-standing, unresolved** upstream issue, not something specific to this repo or this
  install: [vllm-project/vllm#24869](https://github.com/vllm-project/vllm/issues/24869),
  [#19361](https://github.com/vllm-project/vllm/issues/19361) (closed as stale/not-planned,
  unresolved), and matching reports on Hugging Face model discussions for
  [Qwen3-VL-4B](https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit/discussions/1)
  and [gemma-4-31B](https://huggingface.co/unsloth/gemma-4-31B-it-unsloth-bnb-4bit/discussions/1)
  itself (same assertion, same model family as this repo's gemma4 leg).
- [RFC #39583](https://github.com/vllm-project/vllm/issues/39583) confirms bnb support in vLLM
  is low-usage (~0.5%), predates vLLM's current weight-loading architecture
  (`weight_loader_v2`), is full of legacy conditional branches in `linear.py`/`fused_moe`, and
  is being considered for outright removal — consistent with it being fragile/unreliable for
  anything beyond the two basic documented patterns above.

## Verified fix: pre-quantize offline with `transformers`, then load into vLLM as a real pre-quantized checkpoint

Tested end-to-end on the small model:

```python
# 1. Quantize + save with the exact desired bnb settings, using the officially-supported
#    transformers + BitsAndBytesConfig path:
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="cuda:0")
model.save_pretrained(local_dir)  # writes a REAL packed 4-bit checkpoint + correct config.json

# 2. Load the saved checkpoint into vLLM with NO quantization=/hf_overrides at all --
#    vLLM reads quantization_config from local_dir/config.json itself:
llm = LLM(model=local_dir, dtype="bfloat16", ...)
```

**Result: works.** Model loads, CUDA graphs capture, generation succeeds — no shape
assertion, because the checkpoint's weight tensors are genuinely already packed at the
requested precision, matching the parameter shapes vLLM builds from that same config.json.
This is effectively what `build_llm_with_patched_config` was trying to approximate by
patching `config.json` in place, except it symlinked the **original full-precision weight
files** instead of actually re-quantizing them — the config said "4-bit" but the tensors on
disk were still full precision, which is exactly the shape mismatch this whole investigation
chased down.

Separately hit an environment gap while verifying this: no `nvcc` in this container/venv,
which breaks FlashInfer's JIT-compiled sampler on first use (`ninja: build stopped ... nvcc:
not found`). Worked around it with `VLLM_USE_FLASHINFER_SAMPLER=0`. Unrelated to bnb, but
worth checking on the actual run box since it'll affect any vLLM run, not just quantized ones.

## Recommendation

Rework `benchmark/engine.py::build_llm` (and drop `build_llm_with_patched_config`) to:
1. For each `(model, run_entry)` combo with `quant_method == "bitsandbytes"`, load the
   full-precision model via `transformers.AutoModelForCausalLM.from_pretrained(...,
   quantization_config=BitsAndBytesConfig(**run_entry["bnb_args"]))` and `save_pretrained()`
   to a local cache dir (skip if already cached).
2. Point vLLM's `LLM(model=<local cache dir>)` at that directory with no `quantization=` or
   `hf_overrides` — vLLM will infer the quantization from the real config.json.
3. For `quant_method == "none"` rows, load the HF repo directly as today (unaffected).

This keeps vLLM for serving (so throughput/latency numbers stay comparable across rungs)
while sidestepping the broken in-flight-quantization-via-config-override path entirely. It
does add a one-time pre-quantize-and-save step per (model, quant_level) combo — for the two
70B/47B-class models this means an extra full-precision-load-then-quantize pass before the
first vLLM run of that rung, which should be worth budgeting into the run-time estimate in
RUNBOOK.md Phase 3.

## Implemented

`benchmark/engine.py` has been rewritten:
- `build_llm_with_patched_config` and the `hf_overrides`-based quantization branch are
  removed.
- A new `_quantize_and_cache()` helper quantizes a model with `transformers` +
  `BitsAndBytesConfig(**run_entry["bnb_args"])`, saves it to
  `<cache_dir>/<model_id>/<quant_level>/` (default cache dir `/tmp/quantized_models`,
  skipped if already cached), and `build_llm()` points vLLM at that local directory with no
  `quantization=`/`hf_overrides` for any `quant_method == "bitsandbytes"` row.
- `benchmark/run_one_combo.py` and `benchmark/orchestrator.py` had their now-dead
  `--quant-config-mode` flag and `build_llm_with_patched_config` references removed (both
  called it, so the old code would no longer import).

**Smoke-tested end-to-end** against the real rewritten `build_llm()` (not just the manual
verification script above), using the small test model, on both quant levels that
previously failed:
- `int4_nf4_doublequant_bnb` (`bnb_4bit_use_double_quant: true`) — loads, compiles, captures
  CUDA graphs, generates correctly ("Paris" for "capital of France").
- `int8_bnb` (`load_in_8bit: true`) — loads (vLLM correctly falls back to eager mode, since
  bnb 8-bit doesn't support CUDA graphs, matching the RFC's noted limitation), generates
  correctly.

Not yet re-tested: the actual Mixtral/gemma4/Llama-3.3-70B/Qwen2.5-32B legs on their real
checkpoints (only the small stand-in model was used for the fix verification, to avoid
multi-GB re-quantization passes during this investigation). Recommend re-running
`spike_tests/spike_test_mixtral_bnb.py` / `spike_test_gemma4_bnb.py` against this new
`build_llm()` path (or a small adaptation of them) before trusting the full run matrix.
