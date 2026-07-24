# 업데이트: 원래 계획에서 달라진 점

원래 스캐폴딩(`git log`의 초기 커밋)은 단순한 경로를 가정했다: 4개 모델 x N개
quant level에 대해 동일한 bnb 양자화 메커니즘을 그대로 돌려서, 총 14개 combo를
대부분 한 번의 foreground 실행으로 끝낸다는 계획이었다. 실제로 진행된 내용은
여러 지점에서 이 계획과 달라졌고, 그 차이가 조용히 묻히지 않도록 여기에
기록한다.

## 1. 양자화 메커니즘 자체가 교체됨

- **계획**: vLLM의 `hf_overrides={"quantization_config": {...}}`를 통해 모델
  로딩 시점에 즉석으로 양자화를 적용.
- **실제**: 이 메커니즘은 테스트한 모든 vLLM 버전(0.9.2~0.25.1)에서 weight-shape
  `AssertionError`로 크래시하는 것으로 확인됨 (`spike_test_error_report.md`
  참고). 대신 pre-quantize-and-cache 방식으로 교체: `transformers` +
  `BitsAndBytesConfig`로 오프라인에서 미리 양자화하고, 실제 양자화된 체크포인트를
  로컬 캐시 디렉토리에 저장한 뒤, vLLM이 그 체크포인트를 그대로 로드하도록 함
  (`benchmark/engine.py`의 `_quantize_and_cache`). 두 방식 중 하나를 고르던
  `--quant-config-mode` 플래그(`hf_overrides` vs `patched_config`)도 완전히
  제거됨 — 둘 다 같은 이유로 깨져 있었기 때문.

## 2. Mixtral-8x7b가 제외되고, 결국 MoE 자체가 스코프에서 빠짐

- **계획**: 4개 모델(qwen2.5-32b, gemma4-31b, llama3.3-70b, mixtral-8x7b) x
  각 모델의 quant level = 총 14개 combo.
- **실제 (1차)**: Mixtral의 레이어당 8개 MoE expert가 개별 `nn.Linear` 모듈이
  아니라 fused 3D 텐서(`gate_up_proj`/`down_proj`)로 저장되어 있어서,
  bitsandbytes가 이걸 전혀 양자화할 수 없음 — 그래서 Mixtral 파라미터의 약
  96%가 `device_map`, CPU offload, 커스텀 streaming quantizer 어떤 방법을
  써도 전혀 손대지지 않음 (세 방법 모두 검토했지만, 어느 것도 이 구조적
  한계를 해결하지 못함). 처음엔 `configs/run_matrix.yaml`의 `known_risk`
  필드에 알려진 한계로 문서화하고 보류만 함.
- **실제 (2차, 최종)**: "Mixtral 대신 다른 MoE 모델을 테스트할 수 있는지"를
  찾아보기 위해 설치된 transformers 5.14.1 소스코드를 직접 뒤져본 결과,
  Qwen2/3-MoE, OLMoE, GLM4-MoE, DeepSeek-V2/V3, GraniteMoE, PhiMoE, DBRX,
  JetMoE, Cohere2-MoE, Llama4, GPT-OSS 등 **이 버전에 포함된 사실상 모든
  최신 decoder-only MoE 아키텍처가 동일한 fused 3D 텐서 패턴**을 쓰고
  있음을 확인함 (transformers v5의 라이브러리 전체 리팩터링으로 보임).
  bitsandbytes 자체 이슈([#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849))와
  vLLM 이슈([#20480](https://github.com/vllm-project/vllm/issues/20480))에서도
  "BNB는 아직 MoE 모델로 확장되지 않았다"고 명시적으로 확인됨. 즉 Mixtral만의
  문제가 아니라 **이 환경에서 bitsandbytes로 어떤 MoE 모델을 양자화하는 것
  자체가 불가능**하다는 뜻 — 다른 MoE 모델로 바꿔도 해결되지 않음. 이에 따라
  Mixtral을 다시 살리는 대신, **MoE 아키텍처를 이 벤치마크의 스코프에서
  완전히 제외**하기로 결정 (`configs/run_matrix.yaml`에서 mixtral-8x7b 항목
  자체를 삭제, `spike_test_mixtral_bnb.py`도 삭제). 최종 실행은 남은 3개
  모델의 **11개 combo**(qwen2.5-32b 4개 + gemma4-31b 4개 + llama3.3-70b
  3개)만 커버함.

## 3. gemma4-31b의 체크포인트가 잘못되어 있었음

- **계획**: `google/gemma-4-31B`.
- **실제**: 이건 base(instruction-tuning 안 된) 체크포인트라서 chat_template이
  없고, 전체 실행에서 `ChatTemplateResolutionError`로 크래시함. instruction-tuned
  버전인 `google/gemma-4-31B-it`로 교체함.

## 4. `max_model_len`이 전반적으로 너무 짧았음

- **계획**: 기본값 8192, llama3.3-70b는 4096.
- **실제**: Task C(needle-in-a-haystack) 샘플들은 2만~3만 토큰이 필요한데
  둘 다 이보다 훨씬 짧아서, Task C가 실행되자마자 모든 combo가 실패함. 전체를
  32768로 상향함.

## 5. Phase 2 spike test 구조가 바뀜

- `spike_test_mixtral_bnb.py` / `spike_test_gemma4_bnb.py`는 더 이상 raw
  `vllm.LLM(hf_overrides=...)`를 직접 생성하지 않고, `configs/run_matrix.yaml`에서
  `find_entries()`로 실제 `(model_entry, run_entry)` 쌍을 가져와
  `benchmark.engine.build_llm()`을 그대로 호출함 — 그래서 이 spike test의
  PASS/FAIL 결과가 실제 Phase 3의 동작을 그대로 대변함. (`spike_test_mixtral_bnb.py`는
  이후 위 2번 항목에서 설명한 대로 MoE가 스코프에서 완전히 빠지면서 삭제됨.)
- `spike_test_bnb_quant_args.py`는 원래 필수 게이트였지만, 이제는 원래 버그의
  기록으로만 남겨두고 더 이상 실행할 필요 없음.
- `spike_test_build_llm_bnb.py`가 새로 추가됨 — 새 pre-quantize-and-cache
  메커니즘을 작은 대체 모델로 검증하는 스모크 테스트.
- `minimal_bnb_test.py`(git에 추적되지 않던, 같은 깨진 패턴을 가진 중복 스크립트)는
  삭제됨.

## 6. 실행 방식이 바뀜

- **계획**: RUNBOOK의 Phase 3는 한 번에 이어지는 foreground 실행을 전제로 작성됨.
- **실제**: 세션을 여러 시간 동안 계속 열어둘 수 없어서, `nohup ... & disown`으로
  완전히 분리된 백그라운드 프로세스로 실행하고 주기적으로 상태를 확인하는 방식으로
  진행함.

## 7. 계획에 없던 발견: llama3.3-70b의 비단조적 정확도 패턴

원래 계획에는 없던 내용 — 실행이 끝난 뒤 결과를 살펴보다가 발견함.
llama3.3-70b의 Task C(needle-in-a-haystack) 점수가 quant level에 따라
**비단조적**으로 나타남: int8=0.26 -> nf4=0.36 -> nf4+double-quant=0.98,
일반적인 "정밀도가 낮을수록 품질도 낮다"는 트레이드오프와 정반대 방향.
원본 출력을 확인해보니 int8/nf4에서는 모델이 직접 답을 말하는 대신 자주
반복적으로 같은 말을 늘어놓다가 Task의 64토큰 제한에 걸려버리는 반면,
nf4+double-quant는 (qwen처럼) 짧고 직접적으로 답함. 세 체크포인트 모두 이번
실행을 위해 새로 양자화된 것이라 캐시가 오래돼서 생긴 문제는 아님. 이는
양자화로 인한 미세한 수치 노이즈가 greedy(temperature=0) 디코딩을 이 모델
특유의 반복 루프 실패 모드로 밀어넣거나 빠져나오게 만드는 것으로 보이며,
고쳐야 할 버그라기보다는 그 자체로 보고할 가치가 있는 실제 결과로 봐야 함.
`report/report.md`의 known-risks 섹션에 문서화되어 있음.

## 8. Ablation 스터디 추가: "기법이 아니라 조합만 봤다"는 한계를 보완

원래 계획에는 없던, 완주 이후에 추가로 요청받은 확장 작업.

- **동기**: 기존 11개 combo는 전부 "이미 다 조합된 최종 상태"(예:
  `int4_nf4_doublequant_bnb`)만 측정해서, 그 combo의 정확도 하락 중 어디까지가
  weight quantization 때문이고 어디까지가 다른 요인 때문인지 분리할 수 없었음.
  사용자가 제안한 ablation 설계(baseline → weight quant만 → +KV cache quant →
  +activation quant → +CPU offload, 각 축을 최소 accuracy/throughput/VRAM
  3가지로 로깅)를 적용하기로 함.
- **범위 축소 (사용자 결정)**: qwen2.5-32b 1개 모델로만 파일럿, AWQ/GPTQ는
  자체 calibration 없이 커뮤니티(사실은 Qwen팀 공식) 사전 양자화 체크포인트
  사용, accuracy 축에 perplexity(WikiText-2) 추가, CPU offload는 이번
  phase에서 제외.
- **"사다리"가 실제로는 그대로 안 맞음 (중요한 발견)**: 설치된 vLLM 0.25.1
  소스를 직접 읽어서 확인한 결과 —
  - AWQ/GPTQ는 이미 양자화된 체크포인트를 **로드**만 가능 (`quantization=
    "awq_marlin"`/`"gptq_marlin"`), 온더플라이 양자화기는 없음. 그래서
    Qwen팀이 공식 배포한 `Qwen/Qwen2.5-32B-Instruct-AWQ`,
    `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4`를 그대로 사용.
  - KV cache fp8(`kv_cache_dtype="fp8"`)은 정말로 독립적인 축 — calibration
    불필요, 어떤 weight quant 방식과도 결합 가능.
  - **"activation quant"로 가장 간단한 경로인 `quantization="fp8"`은 별도의
    완전한 weight+activation 양자화 방식이지, bnb/AWQ/GPTQ 위에 얹는 레이어가
    아님.** 즉 "weight quant X + activation quant"라는 조합 자체가 이
    툴체인에서는 존재하지 않음 — 사용자가 제안한 4단 사다리 그대로는 구현 불가.
  - 그래서 실제 구현은: "weight quant 방식" 5개(bnb_int8, bnb_int4_nf4_doublequant,
    awq, gptq, fp8_online — 마지막 게 바로 activation quant 역할을 겸함)를
    서로 배타적인 대안으로 두고, 그 위에 `kv_cache_dtype=auto`/`fp8`을 독립
    축으로 얹는 2차원 구조로 재구성함. `configs/ablation_matrix.yaml`,
    `report/report.md`의 methodology에 이 재구성 이유를 명시함.
- **구현**: `configs/ablation_matrix.yaml`(신규, `run_matrix.yaml`은 안 건드림),
  `benchmark/engine.py`에 `kv_cache_dtype` 축 + `awq`/`gptq`/`fp8` quant_method
  분기 추가, `benchmark/task_runners/run_perplexity.py`(신규, vLLM의
  `prompt_logprobs`로 같은 서빙 모델에서 직접 NLL/perplexity 계산 — 별도
  `transformers` forward pass로는 kv_cache_dtype/fp8 축 자체를 측정할 수 없어서
  제외함), `data/prep_scripts/prepare_perplexity_wikitext2.py`(신규),
  `aggregate/schema.py`/`aggregate_results.py`에 `weight_quant_method`/
  `kv_cache_dtype`/`perplexity` 컬럼 추가 (기존 11개 production combo는 fallback
  값으로 하위호환, 재실행 불필요).

## 종합

원래 "동일한 메커니즘으로 4개 모델, 14개 combo를 돌린다"는 계획이, 실제로는
"메커니즘을 통째로 교체하고, 구조적 이유로 모델 1개를 영구히 제외하고, 나머지
3개 모델을 막고 있던 별개의 버그 3개를 고쳐서, 14개 중 11개 combo를 끝까지
완주한다"는 결과로 바뀌었다 — 여기에 더해, 그 자체로 보고할 가치가 있는
계획에 없던 모델 거동 발견이 하나 추가되었고, 완주 이후에는 "combo 단위 비교"의
한계를 보완하기 위한 ablation 스터디(qwen2.5-32b 파일럿)가 별도 파이프라인으로
추가되었다.
