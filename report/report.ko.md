# LLM 양자화 벤치마크 보고서

단일 임대 H100(80GB) GPU에서 vLLM으로 서빙한 3개의 오픈소스 LLM에 대해
여러 bitsandbytes 양자화 수준에 걸쳐 작업 정확도, 처리량(throughput),
메모리 사용량을 비교합니다. MoE 아키텍처(예: Mixtral)는 범위에서
제외됩니다 -- 아래 "알려진 한계" 섹션 참고.

## 방법론

- **모델**: Qwen2.5-32B-Instruct, Llama-3.3-70B-Instruct, google/gemma-4-31B-it
- **양자화**: bitsandbytes를 사용해 `transformers` + `BitsAndBytesConfig`로 오프라인에서 사전 양자화한 뒤 로컬 체크포인트에 저장하고, vLLM이 이를 직접 로드합니다 (vLLM의 즉석(in-flight) `hf_overrides={"quantization_config": ...}` 경로는 테스트한 모든 vLLM 버전(0.9.2~0.25.1)에서 동일하게 재현되는 weight-shape `AssertionError`로 인해 작동하지 않는 것으로 확인됨 -- `docs/spike_test_error_report.md` 참고). Qwen2.5-32B와 gemma-4-31B-it은 FP16/BF16에서 시작하며, Llama-3.3-70B는 FP16 용량(~140GB)이 단일 80GB H100을 초과하기 때문에 INT8에서 시작합니다.
- **작업(Task)**: Task A(도구 사용/JSON 정확성, BFCL, 50개 샘플, key+type 일치 채점), Task B(코드 생성, LiveCodeBench, 2025-01-31 이후 공개된 medium/hard 샘플 50개, 샌드박스 stdin/functional 실행), Task C(Needle in a Haystack, 깊이 0~100%에 걸친 샘플 50개, 정확한 부분 문자열 일치).
- **샘플링**: 모든 생성에 대해 temperature=0(결정론적).
- **지표**: 작업 정확도, 처리량(tokens/sec), TTFT + end-to-end 지연시간, 최대 VRAM 사용량(`nvidia-smi` 폴링으로 측정, `torch.cuda.max_memory_allocated()`와 교차 검증).

- **환경**: NVIDIA H100 NVL (93.1 GB), torch 2.11.0+cu130, vLLM 0.25.1, bitsandbytes 0.49.2.


**중요 유의사항**: Llama-3.3-70B는 VRAM 한계로 인해 (FP16/BF16이 아닌) INT8에서 시작하므로, 동일한 x축 위치라도 모든 모델 간에 직접 비교할 수 없습니다 -- 방법론 참고.

### 데이터 오염(Contamination)

- **Task B (LiveCodeBench)**: `contest_date > 2025-01-31`로 필터링했으며, 이는 3개 대상 모델 중 가장 최근인 `google/gemma-4-31B-it`의 학습 기준일(모델 카드 기준 2025년 1월)보다 명확히 이후입니다. 3개 모델 모두 학습 중에 이 문제들을 접했을 수 없습니다. HumanEval은 2021년부터 공개되어 있어 최신 오픈 LLM들에 의해 상당 부분 암기되었다고 널리 알려져 있으므로 이 때문에 의도적으로 배제했습니다.
- **Task A (BFCL)**: 2024년 2월경 공개. 모델별로 오염 위험이 다릅니다: Llama-3.3-70B의 사전학습 데이터 기준일(모델 카드 기준 2023년 12월 -- 모델 자체는 2024년 12월에 출시되었지만, 기반이 되는 사전학습 코퍼스는 BFCL보다 이전임)은 BFCL 출시 이전이지만, 이후의 instruction-tuning 데이터 수집 과정에서 이론적으로 포함되었을 수 있습니다; Qwen2.5-32B와 gemma-4-31B-it은 BFCL 출시 이후에 학습되었으므로 직간접적으로 노출되었을 수 있습니다. 이러한 비대칭성을 감안하여 Task A의 모델 간 비교를 해석해야 합니다.
- **Task C (Needle in a Haystack)**: Paul Graham 에세이 원문은 3개 모델 모두 거의 확실히 알고 있겠지만, 삽입된 secret 문자열은 데이터 준비 과정마다 새롭게 무작위로 생성되므로 어떤 학습 코퍼스에도 존재할 수 없습니다 -- haystack 원문이 암기되어 있다고 해서 모델이 테스트 대상 정답을 "이미 알고 있는" 것은 아니므로, 이 작업은 상대적으로 오염에 강건합니다.

## 알려진 한계 / 발생한 리스크


11개의 예상 (모델, 양자화 수준) 조합이 모두 이 보고서에 포함되어 있습니다.


**MoE 모델(예: Mixtral-8x7B)은 보완해야 할 누락 조합이 아니라, 이 벤치마크의
범위에서 아예 완전히 제외됩니다**: 현재 사용 중인 transformers 버전에서는
모든 MoE 아키텍처의 expert를 개별 `nn.Linear` 모듈이 아닌 융합된(fused)
3D 텐서(`gate_up_proj`/`down_proj`)로 저장하기 때문에, `device_map`,
CPU 오프로드, 스트리밍 기법과 무관하게 bitsandbytes의 int8/int4 양자화
경로가 이를 전혀 처리할 수 없습니다 -- 이는 Mixtral뿐 아니라 현재
transformers 버전의 사실상 모든 MoE 아키텍처에 영향을 미치는 것으로
확인되었습니다(전체 조사 내용은 `docs/Updates.md`와
`docs/spike_test_error_report.md` 참고).

이 프로젝트 진행 중 사전에 파악되어 추적된 리스크(`configs/run_matrix.yaml`의 `known_risk` 필드, `spike_tests/`, `docs/spike_test_error_report.md` 참고):
- `bnb_4bit_use_double_quant` 전파(propagation)와 gemma-4-31B-it + bitsandbytes 조합은 모두 `spike_tests/spike_test_gemma4_bnb.py`를 통해 end-to-end로 검증되었습니다. 이 스크립트는 손으로 작성한 대체 설정이 아니라, 이번 실행에서 사용한 것과 동일한 코드 경로인 `benchmark.engine.build_llm`을 실제 `configs/run_matrix.yaml`의 행 데이터로 직접 호출합니다.
- **Llama-3.3-70B는 양자화 수준 전반에서 비단조적(non-monotonic)인 정확도 패턴을 보입니다**: Task C(needle-in-a-haystack) 점수가 INT8에서 0.26, NF4에서 0.36인데, NF4+double-quant에서는 0.98로 오히려 가장 정밀도가 낮은 변형이 가장 좋은 성능을 보입니다 -- 일반적인 양자화/품질 트레이드오프와 반대되는 결과입니다. 원본 생성 결과를 살펴보면, INT8/NF4 실행에서는 정답을 바로 제시하지 않고 반복적이고 장황한 텍스트로 빠지는 경우가 잦았고(평균 출력 토큰 약 50~60개, 종종 답변 전에 작업의 64토큰 상한에 도달), NF4+double-quant는 간결하고 직접적으로 답변했습니다(평균 약 10토큰, Qwen과 유사한 스타일). 세 체크포인트 모두 이번 실행을 위해 새로 양자화되었습니다(오래된 캐시 아티팩트가 아님). 이는 단순한 정밀도 대 품질 효과라기보다는, 양자화 노이즈가 그리디(temperature=0) 디코딩을 이 모델 특유의 반복 루프 실패 모드로 비결정론적으로 밀어 넣거나 빠져나오게 만드는 것으로 보입니다 -- Llama-3.3-70B의 Task B/C 수치는 깔끔한 양자화 비교가 아니라 실제로 보고할 가치가 있는 발견으로 취급해야 합니다.

## 결과 테이블 (모델 x 양자화 수준 x 작업별)

| 모델 | 양자화 수준 | 작업 | 정확도 | 샘플 수 | 처리량 (tok/s) | 평균 TTFT (ms) | 평균 E2E (ms) | 최대 VRAM (MB) |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-32b | fp16_baseline | task_a | 0.90 | 50 | 298.4 | nan | nan | 89065 |
| qwen2.5-32b | fp16_baseline | task_b | 0.26 | 50 | 584.9 | nan | nan | 89065 |
| qwen2.5-32b | fp16_baseline | task_c | 1.00 | 50 | 5.1 | nan | nan | 89065 |
| qwen2.5-32b | int8_bnb | task_a | 0.89 | 50 | 81.3 | nan | nan | 88237 |
| qwen2.5-32b | int8_bnb | task_b | 0.18 | 50 | 109.7 | nan | nan | 88237 |
| qwen2.5-32b | int8_bnb | task_c | 1.00 | 50 | 3.4 | nan | nan | 88237 |
| qwen2.5-32b | int4_nf4_bnb | task_a | 0.89 | 50 | 197.1 | nan | nan | 90145 |
| qwen2.5-32b | int4_nf4_bnb | task_b | 0.16 | 50 | 236.6 | nan | nan | 90145 |
| qwen2.5-32b | int4_nf4_bnb | task_c | 1.00 | 50 | 5.0 | nan | nan | 90145 |
| qwen2.5-32b | int4_nf4_doublequant_bnb | task_a | 0.89 | 50 | 196.0 | nan | nan | 90167 |
| qwen2.5-32b | int4_nf4_doublequant_bnb | task_b | 0.16 | 50 | 187.6 | nan | nan | 90167 |
| qwen2.5-32b | int4_nf4_doublequant_bnb | task_c | 1.00 | 50 | 5.0 | nan | nan | 90167 |
| gemma4-31b | bf16_baseline | task_a | 0.84 | 50 | 173.7 | nan | nan | 90033 |
| gemma4-31b | bf16_baseline | task_b | 0.64 | 50 | 520.4 | nan | nan | 90033 |
| gemma4-31b | bf16_baseline | task_c | 1.00 | 50 | 5.3 | nan | nan | 90033 |
| gemma4-31b | int8_bnb | task_a | 0.84 | 50 | 46.9 | nan | nan | 89959 |
| gemma4-31b | int8_bnb | task_b | 0.62 | 50 | 128.8 | nan | nan | 89959 |
| gemma4-31b | int8_bnb | task_c | 1.00 | 50 | 3.3 | nan | nan | 89959 |
| gemma4-31b | int4_nf4_bnb | task_a | 0.84 | 50 | 191.3 | nan | nan | 91527 |
| gemma4-31b | int4_nf4_bnb | task_b | 0.60 | 50 | 362.4 | nan | nan | 91527 |
| gemma4-31b | int4_nf4_bnb | task_c | 1.00 | 50 | 4.9 | nan | nan | 91527 |
| gemma4-31b | int4_nf4_doublequant_bnb | task_a | 0.84 | 50 | 191.2 | nan | nan | 91701 |
| gemma4-31b | int4_nf4_doublequant_bnb | task_b | 0.56 | 50 | 352.8 | nan | nan | 91701 |
| gemma4-31b | int4_nf4_doublequant_bnb | task_c | 1.00 | 50 | 4.9 | nan | nan | 91701 |
| llama3.3-70b | int8_baseline | task_a | 0.83 | 50 | 65.2 | nan | nan | 91157 |
| llama3.3-70b | int8_baseline | task_b | 0.20 | 50 | 31.3 | nan | nan | 91157 |
| llama3.3-70b | int8_baseline | task_c | 0.26 | 50 | 5.1 | nan | nan | 91157 |
| llama3.3-70b | int4_nf4_bnb | task_a | 0.90 | 50 | 94.5 | nan | nan | 93227 |
| llama3.3-70b | int4_nf4_bnb | task_b | 0.34 | 50 | 117.0 | nan | nan | 93227 |
| llama3.3-70b | int4_nf4_bnb | task_c | 0.36 | 50 | 3.2 | nan | nan | 93227 |
| llama3.3-70b | int4_nf4_doublequant_bnb | task_a | 0.90 | 50 | 93.8 | nan | nan | 93277 |
| llama3.3-70b | int4_nf4_doublequant_bnb | task_b | 0.40 | 50 | 124.9 | nan | nan | 93277 |
| llama3.3-70b | int4_nf4_doublequant_bnb | task_c | 0.98 | 50 | 1.9 | nan | nan | 93277 |


## 조합별 요약 (메모리 및 로드 시간, 작업별 아님)

| 모델 | 양자화 수준 | 로드 시간 (s) | 최대 VRAM (MB) | 평균 정확도 (3개 작업) |
|---|---|---|---|---|
| qwen2.5-32b | fp16_baseline | 401.5 | 89065 | 0.72 |
| qwen2.5-32b | int8_bnb | 87.6 | 88237 | 0.69 |
| qwen2.5-32b | int4_nf4_bnb | 151.1 | 90145 | 0.68 |
| qwen2.5-32b | int4_nf4_doublequant_bnb | 152.6 | 90167 | 0.68 |
| gemma4-31b | bf16_baseline | 272.8 | 90033 | 0.83 |
| gemma4-31b | int8_bnb | 135.5 | 89959 | 0.82 |
| gemma4-31b | int4_nf4_bnb | 357.2 | 91527 | 0.81 |
| gemma4-31b | int4_nf4_doublequant_bnb | 362.6 | 91701 | 0.80 |
| llama3.3-70b | int8_baseline | 799.3 | 91157 | 0.43 |
| llama3.3-70b | int4_nf4_bnb | 437.2 | 93227 | 0.53 |
| llama3.3-70b | int4_nf4_doublequant_bnb | 408.0 | 93277 | 0.76 |


## 차트

### 양자화 수준별 정확도 (모델별, 작업별)

![Accuracy vs Quantization Level (per model, per task)](figures/accuracy_vs_quant_per_model.png)

### 양자화 수준별 처리량

![Throughput vs Quantization Level](figures/throughput_vs_quant_per_model.png)

### 양자화 수준별 최대 VRAM

![Peak VRAM vs Quantization Level](figures/memory_vs_quant_per_model.png)

### 처리량 대 정확도 트레이드오프

![Throughput vs Accuracy Trade-off](figures/tradeoff_scatter.png)

### 작업별 정확도 세부 내역

![Per-task Accuracy Breakdown](figures/per_task_breakdown.png)



## 한계점

- Task B 채점은 LiveCodeBench의 공개 테스트 케이스(문제당 2~4개)만 사용하며, 공식 리더보드가 사용하는 더 강력한 held-out 채점용 비공개(인코딩된) 전체 테스트 스위트는 사용하지 않습니다 -- 다소 약하지만 3개 모델 모두에 동일하게 적용되는 모델 중립적인 신호입니다.
- Task A 채점은 공식 BFCL AST-동등성 채점을 재현한 것이 아니라, 프로젝트 자체적으로 단순화한 key/type 매처입니다.
- Llama-3.3-70B의 비교는 FP16/BF16이 아닌 INT8에서 시작합니다 -- Qwen2.5-32B, gemma-4-31B와 동일한 기준선(baseline)에 있지 않습니다.
- Needle-in-a-Haystack의 haystack 토큰 수는 모델 간 일관성을 위해 각 모델의 고유 토크나이저가 아닌 고정된 참조 토크나이저(cl100k_base)를 사용합니다.
- 결과 유효성에 대한 작업별/모델별 유의사항은 위의 데이터 오염 하위 섹션을 참고하세요.
