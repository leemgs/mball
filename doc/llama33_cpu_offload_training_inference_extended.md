모델 학습의 방법중에서 추가로 LoRA/QLoRA 버전 + DeepSpeed ZeRO-3 오프로딩 을 어떻게 하는지에 대한 방법의  예시입니다. 

## 1. Fine-tuning with LoRA / QLoRA + CPU Offload

LoRA(저랭크 어댑터)와 QLoRA(4비트 양자화 LoRA)는
대형 모델을 적은 자원으로 파인튜닝할 수 있는 대표적 기법입니다.
여기서는 Hugging Face [PEFT](https://github.com/huggingface/peft)와 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)를 활용합니다.

```python
# train_lora_cpu_offload.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from accelerate import cpu_offload
from peft import LoraConfig, get_peft_model

MODEL_ID = "meta-llama/Llama-3.3-8B"
EXEC_DEVICE = torch.device("cuda:0")

# 1. Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 2. Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

def tok_fn(examples):
    ids = tok(" ".join(examples["text"]), return_tensors="pt",
              truncation=True, max_length=512)["input_ids"][0]
    return {"input_ids": ids[:-1], "labels": ids[1:]}

proc = dataset.map(tok_fn, remove_columns=dataset.column_names)

# 3. Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,  # QLoRA: requires bitsandbytes
    device_map=None,
)
# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # typical for Llama
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 4. Apply CPU Offload (saves VRAM)
model = cpu_offload(model, execution_device=EXEC_DEVICE, offload_buffers=True)

# 5. Training setup
args = TrainingArguments(
    output_dir="./lora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    bf16=True,
    optim="paged_adamw_32bit",   # bitsandbytes optimizer
)

def collate(batch):
    return {"input_ids": torch.stack([x["input_ids"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch])}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=proc,
    tokenizer=tok,
    data_collator=collate,
)

trainer.train()
trainer.save_model("./lora-checkpoint")
```

---

## 2. DeepSpeed ZeRO-3 Offload Template

`DeepSpeed ZeRO-3`는 **파라미터 + 옵티마이저 상태 + 그래디언트**를 모두
CPU 혹은 NVMe로 분산 저장하여 **대규모 학습에 최적화**된 방식입니다.

아래 예시는 Hugging Face `transformers` + `accelerate` 통합 환경에서
ZeRO-3 Offload 설정을 적용하는 **템플릿**입니다.

### `ds_config_zero3.json`

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9
  },
  "bf16": {
    "enabled": true
  },
  "train_batch_size": 16,
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "wall_clock_breakdown": false
}
```

### 학습 스크립트 (`train_zero3.py`)

```python
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

MODEL_ID = "meta-llama/Llama-3.3-8B"

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

def tok_fn(examples):
    ids = tok(" ".join(examples["text"]), return_tensors="pt",
              truncation=True, max_length=512)["input_ids"][0]
    return {"input_ids": ids[:-1], "labels": ids[1:]}

proc = dataset.map(tok_fn, remove_columns=dataset.column_names)

def collate(batch):
    return {"input_ids": torch.stack([x["input_ids"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch])}

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

args = TrainingArguments(
    output_dir="./zero3-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=50,
    bf16=True,
    deepspeed="./ds_config_zero3.json",  # 핵심
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=proc,
    tokenizer=tok,
    data_collator=collate,
)

trainer.train()
```

**실행 예시**

```bash
accelerate launch train_zero3.py
```

---

## 7. Summary

* **LoRA / QLoRA + cpu\_offload**

  * 경량 학습에 적합
  * VRAM 절약 효과 큼
  * 속도는 다소 저하

* **DeepSpeed ZeRO-3 Offload**

  * 대규모 학습(수십억 파라미터)에 최적
  * CPU/NVMe 오프로딩 가능
  * 학술 논문에서 **대형 모델 학습 재현성 보조 자료**로 권장

---
