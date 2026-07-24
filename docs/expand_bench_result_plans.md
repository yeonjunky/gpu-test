# 다음 벤치마크 실행 계획: 측정 지표 확장

현재 `results.csv` / `results_by_combo.csv`는 "양자화가 VRAM을 줄이는가?"라는
질문에 답할 수 없다. `peak_vram_mb`가 quant level과 거의 무관하게 89~93GB
대역에 몰려있기 때문이다. 원인은 측정 버그가 아니라 `gpu_memory_utilization`이
quant level과 무관하게 모델당 고정값(qwen/gemma 0.90, llama 0.92)으로
vLLM에 전달되고, vLLM이 가중치 로드 후 남는 VRAM을 이 비율까지 KV 캐시로
자동으로 채우기 때문이다 (`benchmark/engine.py`, `configs/run_matrix.yaml`
확인됨). 즉 양자화로 줄어든 가중치 메모리는 사라지는 게 아니라 KV 캐시
용량으로 형태가 바뀌는데, 지금은 그 전환을 보여줄 지표가 없다.

Llama-3.3-70B의 Task C 비단조 정확도(INT8 0.26 → NF4 0.36 → NF4+doublequant
0.98)도 로드 실패나 OOM fallback류의 "깨진 런"이 아니라, raw generation을
직접 조사해서 반복/장황 실패 모드로 이미 원인을 특정한 상태다
(`report.ko.md` "알려진 한계" 섹션 참고). 다만 이걸 다음 런에서 표만 보고
바로 재확인할 수 있는 필드(`run_status`, `seed`, raw output 로깅)가 없다는
점은 실제 공백이다.

## 추가할 필드

### 1. 가중치 절감 효과를 분리해서 보기 위한 필드
- **`weights_vram_mb`** — 모델 로드 직후, 추론 실행 전 `torch.cuda.memory_allocated()`.
  양자화 효과를 순수하게 보여줄 유일한 숫자. 지금 `peak_vram_mb`는 이걸
  가려버린다.
- **`max_memory_reserved_mb`** vs **`max_memory_allocated_mb`** 분리 기록 —
  PyTorch 캐싱 allocator가 reserve만 하고 안 푸는 경우와 실제 텐서 점유를
  구분하기 위함.
- **`num_gpu_blocks`** — vLLM이 실제로 할당한 KV 캐시 블록 수
  (`llm.llm_engine.cache_config`에서 노출). "가중치 절감분이 KV 캐시 용량으로
  전환됐다"는 가설을 직접 증명하는 숫자.
- **`gpu_memory_utilization`**, **`max_model_len`** — 지금 `configs/run_matrix.yaml`에만
  있고 결과 행에는 없음. 재현성과 해석 가능성을 위해 결과 행에도 기록.

### 2. 용량(capacity) 전환 효과를 보기 위한 필드
- 동시 요청 수 또는 최대 배치 크기 — 지금은 단일 요청 기준 tok/s만 측정 중이라
  "양자화로 더 많은 동시 요청을 처리할 수 있다"는 효과가 데이터에 드러나지
  않는다.

### 3. 기존 컬럼을 해석 가능하게 만들기 위한 필드
- **`seq_len`** / **`max_new_tokens`** / **`batch_size`** (task별) — 지금은
  throughput/latency/KV캐시 사용량이 이 값들 없이는 모델 간 비교가 의미 없음.
- **`avg_ttft_ms`**, **`avg_e2e_latency_ms`** — 현재 전 행이 비어 있음(nan).
  실제로 값을 채우거나, 채울 계획이 없다면 컬럼 자체를 제거.
- **`run_status`** / **`error`** / **`seed`** — Llama Task C 같은 이상 현상을
  다음엔 raw generation을 다시 뒤지지 않고 표에서 바로 확인/재현할 수 있게.
- **`gpu_model`** / **`n_gpus`** — 지금 `env_info.json`에 런 전체 기준으로만
  있고 행 단위로는 없음. CSV만 따로 떼어봤을 때 컨텍스트가 빠지는 문제가 있어
  행에도 중복 기록 권장.

### 4. Llama의 fp16/bf16 기준선 부재 문제
물리적으로 80GB에 안 들어가서 전체 추론 기준으로는 fp16/bf16을 못 돌리는 게
맞다. 다만 `weights_vram_mb`만 확보할 목적이라면, 전체 추론 없이 가중치만
fp16으로 로드해보는(또는 멀티 GPU tensor-parallel로 잠깐 로드만 하는) 별도
측정을 추가해서 최소한 "가중치가 실제로 몇 GB 줄었는지"에 대한 기준선은
확보할 수 있다.

## 구현 위치 (예상)
- `weights_vram_mb`, `max_memory_reserved_mb`/`allocated_mb`, `num_gpu_blocks`,
  `run_status`/`error`/`seed`: `benchmark/run_one_combo.py` (현재 `MemoryMonitor`가
  `build_llm()` + 3개 태스크 실행 전체를 하나로 감싸는 지점)
- `gpu_memory_utilization`, `max_model_len`, `gpu_model`, `n_gpus`: 결과 행
  생성 시 `configs/run_matrix.yaml`/`env_info.json` 값을 그대로 복사해 넣기
- `seq_len`/`max_new_tokens`/`batch_size`: 각 task 실행기(`run_task_a/b/c`)
- 동시 요청 수/최대 배치 용량: 별도 부하 테스트 방식 설계 필요 (현재 순차 실행
  구조와는 다른 실행 경로가 필요할 수 있음)

## 우선순위 제안
1. `weights_vram_mb` — 가장 시급함. 이게 없으면 "양자화가 VRAM을 줄인다"는
   핵심 주장을 다음 런에서도 증명할 수 없음.
2. `run_status`/`error`/`seed` — Llama 이상 현상 재현성 확보.
3. `num_gpu_blocks`, `gpu_memory_utilization`, `max_model_len` — 절감분이
   KV 캐시로 전환된다는 가설의 직접 증거.
4. `seq_len`/`batch_size`/`avg_ttft_ms`/`avg_e2e_latency_ms` 정리.
5. 동시 요청 용량 측정 — 가장 손이 많이 감, 별도 설계 필요.
