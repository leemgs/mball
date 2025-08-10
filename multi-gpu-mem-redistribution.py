import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time

def setup(rank, world_size):
    """ 분산 환경 설정 (멀티 GPU 설정) """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """ 분산 환경 종료 """
    dist.destroy_process_group()

class MBALLDistributed:
    def __init__(self, model_name='gpt2', rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.model_name = model_name
        self.device = torch.device(f"cuda:{rank}")
        self.setup()
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model = DDP(self.model, device_ids=[rank])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def setup(self):
        """ 멀티 GPU 환경 설정 """
        setup(self.rank, self.world_size)

    def load_data(self):
        """ 데이터를 로딩하는 함수 (예시로 간단한 텍스트 데이터 사용) """
        input_text = ["The future of AI is", "The evolution of machine learning."]
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        return inputs["input_ids"].to(self.device)

    def infer(self, input_ids):
        """ 모델 추론 """
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.module.generate(input_ids, max_length=50)
        inference_time = time.time() - start_time

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Inference result on GPU {self.rank}: {output_text}")
        print(f"Inference time on GPU {self.rank}: {inference_time:.2f} seconds")

    def cleanup(self):
        """ 클린업: 분산 환경 종료 """
        cleanup()

def main(rank, world_size):
    """ 분산 학습/추론을 위한 메인 함수 """
    # MBALLDistributed 클래스 인스턴스화
    model_instance = MBALLDistributed(model_name="gpt2", rank=rank, world_size=world_size)

    # 데이터 로딩
    input_ids = model_instance.load_data()

    # 추론 수행
    model_instance.infer(input_ids)

    # 클린업
    model_instance.cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 사용 가능한 GPU 수
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
