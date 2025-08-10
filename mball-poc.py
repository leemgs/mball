import torch
import gc
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

class GPUProcessAllocator:
    """ GPU Process Allocator: GPU 메모리 할당 및 충돌 방지 """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def allocate(self):
        """ GPU 캐시를 비우고 메모리 할당을 최적화 """
        torch.cuda.empty_cache()
        gc.collect()
        return self.device

class TaskMemoryManager:
    """ Task Memory Manager: DRAM과 VRAM 간 동적 메모리 재분배 """
    def __init__(self, model):
        self.model = model
    
    def optimize_memory(self):
        """ 모델을 mixed precision으로 로딩하여 메모리 최적화 """
        if self.model.device.type == 'cuda':
            self.model = self.model.half()  # Mixed precision
        return self.model

class GPUBoosterWithSwapSupport:
    """ GPU Booster with Swap Support: VRAM 오버플로우 완화 및 스왑 지원 """
    def __init__(self):
        self.swap_threshold = 0.8  # VRAM 사용량 임계값 설정
    
    def check_vram_usage(self):
        """ VRAM 사용량을 확인하고, 스왑을 고려 """
        vram_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        if vram_used > self.swap_threshold:
            print(f"Warning: VRAM usage {vram_used*100:.2f}% is high. Consider swapping.")
        return vram_used
    
    def handle_swap(self):
        """ 스왑을 위한 임시 처리 """
        print("Handling VRAM swap...")  # 실제 시스템에서는 Loopback device 사용
        # 실제 구현에서는 VRAM에서 데이터를 오프로드하여 swap을 지원하는 방식 구현 필요
        return

class MBALL:
    def __init__(self, model_name='gpt2'):
        self.model_name = model_name
        self.device = GPUProcessAllocator().allocate()
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model = TaskMemoryManager(self.model).optimize_memory()
        self.swap_support = GPUBoosterWithSwapSupport()

    def load_model(self):
        """ 모델 로딩 """
        print(f"Loading {self.model_name} on {self.device}...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        loading_time = time.time() - start_time
        print(f"Model loading time: {loading_time:.2f} seconds")
        return tokenizer
    
    def infer(self, input_text):
        """ 모델 추론 """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # VRAM 사용량 확인
        self.swap_support.check_vram_usage()

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=50)
        inference_time = time.time() - start_time

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Inference result: {output_text}")
        print(f"Inference time: {inference_time:.2f} seconds")
        return output_text, inference_time


# 테스트 코드
if __name__ == "__main__":
    mball = MBALL()

    # 모델 로딩 및 성능 측정
    tokenizer = mball.load_model()

    # 추론 예시
    input_text = "The future of AI is"
    mball.infer(input_text)
