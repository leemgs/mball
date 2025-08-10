import torch
import time
import gc
import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

# MBALL의 핵심 구성요소를 위한 기본 함수들

def gpu_process_allocator():
    """ GPU Process Allocator: GPU 메모리 관리 및 충돌 방지 """
    torch.cuda.empty_cache()

def task_memory_manager(model):
    """ Task Memory Manager: DRAM과 VRAM 간의 메모리 재분배 최적화 """
    # 예시로 모델을 메모리 최적화 설정으로 로딩
    model = model.half()  # Mixed precision 사용
    return model

def gpu_booster_with_swap_support(model):
    """ GPU Booster with Swap Support: VRAM 오버플로우를 완화 """
    # GPU 메모리가 부족할 경우 스왑을 지원하는 구조를 가정
    if torch.cuda.memory_allocated() > torch.cuda.get_device_properties(0).total_memory * 0.9:
        # Swap 사용을 위한 최적화 예시 (가상 환경에 맞게 조정)
        print("Low VRAM, consider swapping.")
    return model

# LLM 모델 로딩 및 추론 테스트 함수
def load_and_infer_model(model_name='gpt2', use_mball=False):
    start_time = time.time()

    # 모델 로딩
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # MBALL 적용 여부에 따른 최적화
    if use_mball:
        model = gpu_process_allocator()
        model = task_memory_manager(model)
        model = gpu_booster_with_swap_support(model)

    # 모델 로딩 시간 측정
    loading_time = time.time() - start_time
    print(f"Model loading time: {loading_time:.2f} seconds")

    # 모델 추론 예시
    input_text = "The future of AI is"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(torch.device("cuda"))

    # 추론 시간 측정
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50)
    inference_time = time.time() - start_time

    # 추론 결과 및 성능 측정
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Inference result: {output_text}")
    print(f"Inference time: {inference_time:.2f} seconds")

    return loading_time, inference_time

# POC 실행
if __name__ == "__main__":
    model_name = 'gpt2'
    
    # MBALL을 적용하지 않은 경우
    print("Testing without MBALL:")
    loading_time_no_mball, inference_time_no_mball = load_and_infer_model(model_name=model_name, use_mball=False)

    # MBALL을 적용한 경우
    print("\nTesting with MBALL:")
    loading_time_mball, inference_time_mball = load_and_infer_model(model_name=model_name, use_mball=True)

    # 성능 비교
    print(f"\nPerformance Improvement with MBALL:")
    print(f"Model Loading Time (MBALL vs No MBALL): {loading_time_no_mball:.2f}s vs {loading_time_mball:.2f}s")
    print(f"Inference Time (MBALL vs No MBALL): {inference_time_no_mball:.2f}s vs {inference_time_mball:.2f}s")
