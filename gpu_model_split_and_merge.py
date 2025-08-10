import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class ModelParallelLayer(nn.Module):
    """ 모델의 특정 층을 담당하는 파라미터 분할 """
    def __init__(self, layer, device):
        super(ModelParallelLayer, self).__init__()
        self.layer = layer
        self.device = device

    def forward(self, x):
        # 입력을 해당 GPU로 이동
        x = x.to(self.device)
        # 모델 층을 GPU에서 처리
        x = self.layer(x)
        return x

class ModelParallel(nn.Module):
    """ 모델을 여러 GPU에 분할하여 처리 """
    def __init__(self, model_name='gpt2', num_gpus=2):
        super(ModelParallel, self).__init__()
        self.num_gpus = num_gpus
        self.device_ids = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        
        # 모델 로딩
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 모델의 특정 층을 각 GPU에 분배하여 파라미터 분할
        self.layers = nn.ModuleList()
        total_layers = len(self.model.transformer.h)
        
        layers_per_gpu = total_layers // self.num_gpus
        for i in range(self.num_gpus):
            start_layer = i * layers_per_gpu
            end_layer = (i + 1) * layers_per_gpu if i != self.num_gpus - 1 else total_layers
            device = self.device_ids[i]
            
            # 특정 GPU에서 처리할 모델 층을 할당
            parallel_layers = nn.ModuleList(self.model.transformer.h[start_layer:end_layer])
            self.layers.append(ModelParallelLayer(parallel_layers, device))

    def forward(self, input_ids):
        x = input_ids
        # 각 GPU에서 계산
        for i in range(self.num_gpus):
            x = self.layers[i](x)
        return x

class MBALLParallel:
    """ 멀티 GPU에서 모델 병렬화 및 추론 수행 """
    def __init__(self, model_name='gpt2', num_gpus=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelParallel(model_name=model_name, num_gpus=num_gpus
