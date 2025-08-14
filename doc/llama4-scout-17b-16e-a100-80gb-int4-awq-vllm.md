# Requirement for Evaluation
* 조건(배치 1, 컨텍스트 8–16k, **INT8**)
* 검토: **Scout 17B-16E를 A100 80GB 단일 카드에서 사실상 구동 불가**입니다.
이유는 MoE 특성상 **모든 전문가 가중치가 VRAM에 상주시** 해야 하는데, 4bit에서도 **\~60GB**를 먹는 리포트가 있고(→ 8bit이면 단순비례로 **\~120GB+**), 여기에 KV 캐시가 추가되면 80GB를 크게 초과합니다. ([Reddit][1])


아래는 **예상 VRAM 산정(배치=1)** 과 **실행 커맨드(현실적 대안 포함)** 입니다.

---

# 1) 예상 VRAM (INT8 가정)

## 가중치(Weights)

* 경험칙: 4bit 적재가 \~60GB → **8bit ≈ \~120GB**(+ 스케일/메타데이터 오버헤드 약간).
  ⇒ **80GB 초과로 불가**. ([Reddit][1])

## KV 캐시

* 근사식: `KV(bytes) ≈ 2 × n_layers × hidden_dim × bytes_per_val × tokens` (토큰당) (FP16=2B, FP8=1B 등). ([instinct.docs.amd.com][2])
* 직관적 근사: **\~1MB/토큰(13B 기준)** ⇒ 17B 계열은 **\~1.1–1.3MB/토큰**으로 가정. ([rohan-paul.com][3])

| 설정   | 토큰수 |    KV(FP16) 근사 |     KV(FP8) 근사\* |
| ---- | --: | -------------: | ---------------: |
| 배치 1 |  8k |  **\~9–11 GB** | **\~4.5–5.5 GB** |
| 배치 1 | 16k | **\~18–22 GB** |    **\~9–11 GB** |

\* vLLM의 **FP8 KV 캐시** 지원으로 반감 가능. ([VLLM Docs][4])

> 결론: **INT8 가중치만으로도 80GB 초과**이며, KV는 그 위에 추가로 필요하므로 **단일 A100 80GB에서 INT8은 불가**.

---

# 2) 정확 실행 커맨드 (현실적 대안 2가지)

## A. **단일 A100 80GB에서 확실히 되는 경로** — **INT4(AWQ/GPTQ) + FP8 KV**

vLLM이 Llama-4(Scout/Maverick) **Day-0 지원**하며, MoE에 대해 **BNB 4bit는 미지원** 이슈가 있으니 **AWQ/GPTQ 4bit**를 권장합니다. ([vLLM Blog][5], [vLLM Forums][6], [GitHub][7])

```bash
# vLLM 최신 설치 (예: 0.8.3+)
pip install -U "vllm>=0.8.3" autoawq transformers

# 4bit(AWQ) 모델을 준비해 두었다고 가정
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Scout-17B-16E-Instruct-AWQ-4bit \
  --quantization awq \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --dtype auto \
  --enqueue-delay 0
```

> 메모: 8k 컨텍스트/배치 1은 여유, 16k는 KV가 늘어 **배치 1 유지** 권장. ([vLLM Blog][5])

간단 테스트:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Scout-17B-16E-Instruct-AWQ-4bit",
       "messages":[{"role":"user","content":"Hello!"}],
       "temperature":0.2}'
```

## B. **정말 INT8을 원할 때** — **멀티 GPU(≥2×A100 80GB)**

단일 80GB로는 불가하므로 텐서 병렬로 쪼개야 합니다. (아래는 예시; 실제 필요 VRAM과 토폴로지에 맞게 조정)

```bash
# 2장 이상 필요 (예: 2×A100 80GB)
pip install -U "vllm>=0.8.3"

# 사전 8bit 양자화(GPTQ 8bit) 가중치가 준비돼 있다고 가정
# 주의: BNB 8bit 경로는 MoE에서 제약 있음. AWQ/GPTQ 사용 권장.  :contentReference[oaicite:7]{index=7}
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Scout-17B-16E-Instruct-GPTQ-8bit \
  --quantization gptq \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --kv-cache-dtype fp16 \
  --dtype auto
```

---

# 3) 참고 근거

* **vLLM Llama-4 지원 공지(0.8.3+)**. ([vLLM Blog][5])
* **Scout 모델 카드:** BF16 배포 + **on-the-fly INT4** 코드 안내(단일 GPU 현실 경로). ([Hugging Face][8])
* **vLLM MoE에서 BNB(INT4/8) 제약**(FusedMoE 커널 이슈) — AWQ/GPTQ 권장. ([vLLM Forums][6], [GitHub][7])
* **KV 캐시 공식/근사식 & FP8 KV**. ([instinct.docs.amd.com][2], [VLLM Docs][4])
* **4bit에서 \~60GB 사용 사례(경험치)** → 8bit시 단순비례로 120GB+. ([Reddit][1])

---

## 요약

* **INT8(배치 1, 8–16k)**: **단일 A100 80GB 불가** (가중치만 \~120GB+).
* **실전 선택지**: **INT4(AWQ/GPTQ) + FP8 KV + vLLM(≥0.8.3)** → 단일 80GB에서 OK.
* 더 큰 배치/긴 컨텍스트/INT8 고집 시: **2장 이상**으로 텐서 병렬.

[1]: https://www.reddit.com/r/LocalLLaMA/comments/1mf6gaa/llama4scout17b16einstructggufq4_k_s_running_at_20/?utm_source=chatgpt.com "Llama-4-Scout-17B-16E-Instruct-GGUF:Q4_K_S running at ..."
[2]: https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/run-vllm.html?utm_source=chatgpt.com "Steps to Run a vLLM Workload - AMD Instinct Docs"
[3]: https://www.rohan-paul.com/p/kv-caching-in-llm-inference-a-comprehensive?utm_source=chatgpt.com "KV Caching in LLM Inference A Comprehensive Review"
[4]: https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html?utm_source=chatgpt.com "Quantized KV Cache - vLLM"
[5]: https://blog.vllm.ai/2025/04/05/llama4.html?utm_source=chatgpt.com "Llama 4 in vLLM"
[6]: https://discuss.vllm.ai/t/moe-quantization/594?utm_source=chatgpt.com "MoE quantization - vLLM Forums"
[7]: https://github.com/vllm-project/vllm/issues/20480?utm_source=chatgpt.com "BitsandBytes Quantization Support for MoE Models · Issue ..."
[8]: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct?utm_source=chatgpt.com "meta-llama/Llama-4-Scout-17B-16E-Instruct"
