import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class GPUBoosterWithSwapSupport:
    """ GPU Booster with Swap Support: VRAM 오버플로우 완화 및 스왑 지원 """
    def __init__(self, swap_threshold=0.8, swap_device="/tmp"):
        self.swap_threshold = swap_threshold  # VRAM 사용량 임계값
        self.swap_device = swap_device  # 스왑을 위한 디스크 위치
        self._ensure_swap_device()
    
    def _ensure_swap_device(self):
        """ 스왑 디바이스가 존재하는지 확인하고 없으면 생성 """
        if not os.path.exists(self.swap_device):
            os.makedirs(self.swap_device)
    
    def check_vram_usage(self):
        """ VRAM 사용량을 확인하고, 스왑을 고려 """
        vram_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM Usage: {vram_used*100:.2f}%")
        if vram_used > self.swap_threshold:
            print(f"Warning: VRAM usage {vram_used*100:.2f}% is high. Initiating swap...")
            self.handle_swap()
    
    def handle_swap(self):
        """ VRAM이 부족할 경우 데이터를 디스크로 오프로드하여 VRAM을 확장 """
        # 실제 VRAM에서 데이터를 오프로드하여 디스크로 스왑하는 로직 (가상 예시)
        print("Swapping data to disk...")

        # 예시: 텐서를 CPU로 이동시켜 VRAM을 확보하는 방법
        # (실제 구현에서는 디스크 I/O를 최적화하여 스왑을 처리)
        for obj in torch.cuda.memory_cached():
            torch.cuda.synchronize()
            # 오프로드 처리 예시
            obj.cpu()  # CPU로 이동
            print("Data swapped to CPU (disk emulation).")
        
        # 추가적인 스왑 처리 방법 구현 필요: Loopback Block Device 등

    def swap_vram_to_cpu(self, tensor):
        """ VRAM에서 CPU로 데이터를 이동시켜 VRAM을 확장 """
        print(f"Swapping tensor {tensor.size()} to CPU...")
        tensor_cpu = tensor.cpu()
        return tensor_cpu


class MBALL:
    def __init__(self, model_name='gpt2'):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.swap_support = GPUBoosterWithSwapSupport()

    def load_model(self):
        """ 모델 로딩 """
        print(f"Loading {self.model_name} on {self.device}...")
        start_time = time.time()
        loading_time = time.time() - start_time
        print(f"Model loading time: {loading_time:.2f} seconds")
        return self.tokenizer
    
    def infer(self, input_text):
        """ 모델 추론 """
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # VRAM 사용량 확인
        self.swap_support.check_vram_usage()

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=50)
        inference_time = time.time() - start_time

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
