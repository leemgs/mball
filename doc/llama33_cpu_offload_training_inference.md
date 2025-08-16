아래는 \*\*`accelerate.cpu_offload`\*\*를 이용해 **Llama 3.3**(가칭)을 **학습(파인튜닝) / 추론**하는 최소 예제입니다.

> ⚠️ 주의: `cpu_offload`는 본래 **추론에 최적**이고, 학습에는 성능 저하가 큽니다. 그래도 “돌아가는” 기준의 미니멀 예제를 드립니다. 실제 학습에선 **DeepSpeed ZeRO / FSDP+CPU offload**가 훨씬 효율적입니다.

---

# 1) 학습 예제 (미니멀 파인튜닝)

* 토이 데이터셋(wikitext-2)로 causal LM 파인튜닝
* 모델 가중치는 CPU에 상주, **forward/backward 시에만 GPU로 이동**
* 입력 텐서는 **GPU로 올려야** 합니다(execution\_device와 동일)

```python
# train_cpu_offload.py
import os
import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from accelerate import cpu_offload

# ===== 사용자 설정 =====
# 실제 사용 시 공개 모델 ID를 정확히 넣으세요 (예: "meta-llama/Llama-3.3-8B")
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.3-8B")
EXEC_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
BATCH_SIZE = 1          # VRAM 절약용
GRAD_ACCUM = 8
LR = 2e-5
NUM_EPOCHS = 1
MAX_STEPS = 100         # 데모 목적. 실제는 늘리세요.
BLOCK_SIZE = 1024

assert EXEC_DEVICE.type == "cuda", "cpu_offload의 execution_device는 보통 GPU여야 합니다(추론/학습 속도 모두)."

# ===== 토크나이저/데이터 =====
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")  # 아주 작은 샘플

def tok_fn(examples):
    text = examples["text"]
    # 간단히 이어붙여 학습용 블록을 만듭니다.
    joined = "\n\n".join(text)
    ids = tok(joined, return_tensors="pt", truncation=True, max_length=BLOCK_SIZE)["input_ids"][0]
    # 다음 토큰 예측을 위해 labels=inputs를 그대로 사용
    return {"input_ids": ids[:-1], "labels": ids[1:]}

proc = raw.map(lambda ex: tok_fn(ex), remove_columns=raw.column_names)
# 텐서 배치화 (각 샘플이 이미 1개 시퀀스)
def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])    # [B, T]
    labels    = torch.stack([b["labels"]    for b in batch])    # [B, T]
    return {"input_ids": input_ids, "labels": labels}

loader = DataLoader(proc, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

# ===== 모델 로드 & CPU 오프로딩 =====
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    device_map=None,                # 반드시 None (accelerate가 이동 제어)
)

# 학습 시 메모리 절약에 도움
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# 핵심: 가중치는 CPU에 두고, 실행 시에만 cuda:0으로 이동/복귀
model = cpu_offload(
    model,
    execution_device=EXEC_DEVICE,
    offload_buffers=True,           # 버퍼도 오프로드
)

# 옵티마이저/스케줄러 (옵티마이저 상태는 기본적으로 파라미터가 있는 곳과 함께 관리됩니다)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
num_update_steps_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
max_train_steps = min(MAX_STEPS, NUM_EPOCHS * num_update_steps_per_epoch)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, max_train_steps // 10),
    num_training_steps=max_train_steps,
)

scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE in (torch.float16, torch.bfloat16)))

model.train()
global_step = 0
optimizer.zero_grad(set_to_none=True)

for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(loader):
        if global_step >= max_train_steps:
            break

        # 입력은 execution_device로!
        input_ids = batch["input_ids"].to(EXEC_DEVICE, non_blocking=True)
        labels    = batch["labels"].to(EXEC_DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=DTYPE, enabled=True):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            global_step += 1

            if global_step % 10 == 0:
                print(f"step {global_step}/{max_train_steps} | loss={out.loss.item():.4f}")

        # 🔎 데모 목적이라 약간의 해제 타이밍 여유를 줌(선택)
        torch.cuda.synchronize()

    if global_step >= max_train_steps:
        break

# 체크포인트 저장(가중치는 CPU에 있으므로 일반 저장 가능)
save_dir = "./llama33-offload-checkpoint"
model.save_pretrained(save_dir)
tok.save_pretrained(save_dir)
print("Saved to", save_dir)
```

**실행 예시**

```bash
pip install "transformers>=4.42" "accelerate>=0.30" datasets torch --upgrade
CUDA_VISIBLE_DEVICES=0 MODEL_ID=meta-llama/Llama-3.3-8B python train_cpu_offload.py
```

> 팁
>
> * 배치/시퀀스 길이를 낮추고 `GRAD_ACCUM`으로 수렴을 노리세요.
> * 너무 느리면 **ZeRO-2/3 Offload** 또는 **FSDP + CPU Offload**로 전환 권장.

---

# 2) 추론 예제 (`generate`) — CPU 오프로딩

```python
# infer_cpu_offload.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import cpu_offload

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.3-8B")
EXEC_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16

assert EXEC_DEVICE.type == "cuda", "추론에서도 execution_device는 GPU 사용을 권장합니다."

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    device_map=None,            # accelerate가 이동 관리
)
# 필요 시
if hasattr(model, "eval"):
    model.eval()

# 핵심: CPU 오프로딩 적용
model = cpu_offload(
    model,
    execution_device=EXEC_DEVICE,
    offload_buffers=True,
)

prompt = """You are Llama 3.3. Briefly explain CPU offloading vs disk offloading for inference."""
inputs = tok(prompt, return_tensors="pt")
inputs = {k: v.to(EXEC_DEVICE, non_blocking=True) for k, v in inputs.items()}

with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE, enabled=True):
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

print(tok.decode(out[0], skip_special_tokens=True))
```

**실행 예시**

```bash
pip install "transformers>=4.42" "accelerate>=0.30" torch --upgrade
CUDA_VISIBLE_DEVICES=0 MODEL_ID=meta-llama/Llama-3.3-8B python infer_cpu_offload.py
```

---

## 성능/안정성 팁

* **입력 텐서**는 항상 `execution_device`(예: `cuda:0`)로 올리세요.
* `offload_buffers=True`로 버퍼도 내리면 VRAM을 조금 더 아낄 수 있습니다.
* **BF16**이 가능하면 BF16 권장(H100/A100 등), 아니라면 FP16.
* 속도가 너무 느리면 \*\*양자화(AWQ/GPTQ/BNB)\*\*와 병행하거나, \*\*`cpu_offload_with_hook`\*\*로 파이프라인 교대 상주를 설계하세요.
* **학습**을 본격적으로 할 땐 `accelerate` 단독 오프로딩 대신 **DeepSpeed ZeRO-3 Offload** 또는 **PyTorch FSDP(+CPU offload)** 구성이 정석입니다.

