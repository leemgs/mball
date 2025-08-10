import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

class MemoryOptimizer:
    """메모리 최적화: DRAM과 VRAM 간의 메모리 재분배 및 최적화"""
    
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def optimize_memory(self):
        """모델을 mixed precision으로 최적화하여 메모리 절약"""
        if self.device.type == 'cuda':
            self.model = self.model.half()  # Mixed precision 사용 (16-bit)
        return self.model
    
    def swap_memory(self, tensor):
        """메모리를 VRAM에서 DRAM(또는 CPU)으로 재분배"""
        if torch.cuda.memory_allocated() > torch.cuda.get_device_properties(0).total_memory * 0.8:
            print("VRAM 사용량이 높습니다. DRAM으로 데이터를 이동시킵니다.")
            tensor_cpu = tensor.cpu()  # GPU에서 CPU로 텐서 이동
            return tensor_cpu
        return tensor

    def clear_cache(self):
        """GPU 캐시 비우기"""
        torch.cuda.empty_cache()
        gc.collect()

class MBALL:
    def __init__(self, model_name='gpt2'):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.memory_optimizer = MemoryOptimizer(self.model)

    def load_model(self):
        """모델 로딩"""
        print(f"Loading {self.model_name} on {self.device}...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        loading_time = time.time() - start_time
        print(f"Model loading time: {loading_time:.2f} seconds")
        return tokenizer

    def infer(self, input_text):
        """모델 추론"""
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # 메모리 최적화: VRAM과 DRAM 간 동적 재분배
        input_ids = self.memory_optimizer.swap_memory(input_ids)

        # Mixed precision을 통해 메모리 최적화
        self.model = self.memory_optimizer.optimize_memory()

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
