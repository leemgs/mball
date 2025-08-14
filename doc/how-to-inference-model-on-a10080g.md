# meta-llama/Llama-4-Scout-17B-16E-Instruct 모델을 Nvidia A100 GPU VRAM 80GB에서 실행할려면?

A100 80GB 한 장으로 가능합니다. 다만 FP16/BF16은 거의 확실히 안 맞고, 4bit(권장)·8bit 양자화로 서빙해야 합니다. 
* vLLM ≥ 0.8.3 + **AWQ/GPTQ 4bit**가 가장 실용적입니다.
* (BNB 4bit는 MoE 커널 이슈로 vLLM에서 아직 비권장) ([vLLM Blog][1], [vLLM Forums][2])

---

## 무엇이 가능한가?

* **단일 A100 80GB + INT4(AWQ/GPTQ)** → O, 가능. Scout는 MoE(16E) 구조지만 4bit로 적재 시 **\~60GB대** VRAM을 요구한다는 리포트가 다수라, 8–16k 컨텍스트·배치 1\~2면 80GB에 들어옵니다. ([Reddit][3])
* **BF16/FP16 단일 A100** → x, 사실상 불가(체크포인트가 80GB를 크게 초과: 커뮤니티 기준 BF16 **\~220GB** 수준). ([Reddit][3])
* **FP8** → x, A100은 FP8 가속 미지원(이 경로는 H100/B200 대상). ([NVIDIA Developer][4], [NVIDIA GitHub][5])
* **vLLM 지원 상태** → Llama-4(Scout/Maverick) **Day-0 지원** 공지. 단, 일부 이슈에서 Scout **BF16은 4×80GB**, INT4 작업 진행 중 등 레퍼런스도 있어 최신 vLLM 사용 권장. ([vLLM Blog][1], [GitHub][6])
* **모델 카드 참고** → 공식 HF 카드에 Scout는 **BF16 배포**, **on-the-fly INT4 양자화** 코드 제공 언급. 단일 GPU 운용은 **INT4 전제**가 현실적. ([Hugging Face][7])

---

## 권장 실행 절차(A100 80GB, vLLM + INT4/AWQ)

1. **허깅페이스 접근 허용**

   * `meta-llama/Llama-4-Scout-17B-16E-Instruct` 라이선스 동의 및 토큰 준비. ([Hugging Face][8])

2. **환경**

   * Python 3.10+, CUDA 12.x
   * `pip install -U vllm>=0.8.3 autoawq transformers`  (vLLM 최신 필수) ([vLLM Blog][1])

3. **양자화(AWQ)**

   * BF16 체크포인트 → **AWQ 4bit**로 오프라인 양자화 (권장: group-size 128, symmetric, per-channel).
   * 이유: **vLLM은 MoE에 대해 BNB INT4 미지원** 이슈가 있어 **AWQ/GPTQ** 경로가 안정적. ([vLLM Forums][2])

4. **서빙 실행(vLLM)**

   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model /path/to/Scout-17B-16E-Instruct-AWQ-4bit \
     --quantization awq \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.92 \
     --dtype auto \
     --enqueue-delay 0
   ```

   * 처음엔 **max-model-len=8k, 배치=1**로 시작 → 여유 확인 후 확장.
   * 긴 컨텍스트(≥32k)는 **KV 캐시 메모리**가 급증하므로 배치를 1로 유지하거나 멀티-GPU로 확장. ([vLLM Blog][1])

5. **간단 테스트**

   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Scout-17B-16E-Instruct-AWQ-4bit",
          "messages":[{"role":"user","content":"Hello!"}],
          "temperature":0.2}'
   ```

---

## 메모리 감 잡기(단일 80GB)

* **가중치**: Scout는 총 109B(16E)지만 4bit 양자화 시 **가중치 적재 \~60GB대** 보고. 남는 VRAM은 **KV 캐시/런타임 버퍼**에 사용됩니다. 8k×1에서 여유, 16k×1도 보통 수용 가능. ([Reddit][3])
* **FP16/BF16**: 단일 80GB에 **미수용** → 분산(예: 4×80GB) 또는 오프로딩 필요. ([GitHub][6])

---

## 대안 경로(옵션)

* **TensorRT-LLM**: H100/B200에서 **FP8** 경로가 매우 빠름. A100은 FP8 미지원이라 실익 낮음. ([NVIDIA Developer][4], [NVIDIA GitHub][5])
* **멀티-GPU(A100×2 이상)**: 더 긴 컨텍스트/배치가 필요하면 텐서 병렬로 확장.

---

원하시면 **배치/컨텍스트/양자화(4/8bit)** 조건을 주시면, 그 값으로 **예상 VRAM(가중치+KV 캐시) 계산**과 함께 **정확 실행 커맨드**를 바로 맞춰 드릴게요.

[1]: https://blog.vllm.ai/2025/04/05/llama4.html?utm_source=chatgpt.com "Llama 4 in vLLM"
[2]: https://discuss.vllm.ai/t/moe-quantization/594?utm_source=chatgpt.com "MoE quantization - vLLM Forums"
[3]: https://www.reddit.com/r/LocalLLaMA/comments/1jsrz5v/109b_vs_24b_whats_this_benchmark/?utm_source=chatgpt.com "109b vs 24b ?? What's this benchmark? : r/LocalLLaMA"
[4]: https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/?utm_source=chatgpt.com "NVIDIA Accelerates Inference on Meta Llama 4 Scout and ..."
[5]: https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html?utm_source=chatgpt.com "H100 has 4.6x A100 Performance in TensorRT-LLM ..."
[6]: https://github.com/vllm-project/vllm/issues/16197?utm_source=chatgpt.com "Not able to deploy Llama-4-Scout-17B-16E-Instruct on vllm ..."
[7]: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct?utm_source=chatgpt.com "meta-llama/Llama-4-Scout-17B-16E-Instruct"
[8]: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E/tree/main?utm_source=chatgpt.com "meta-llama/Llama-4-Scout-17B-16E at main"


