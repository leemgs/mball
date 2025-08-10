import torch
import os
import numpy as np
import time
from torch import nn

def set_numa_affinity(cpu_id):
    """ 주어진 CPU ID에 NUMA 노드 할당 """
    os.sched_setaffinity(0, {cpu_id})  # CPU 친화성 설정
    print(f"CPU {cpu_id} affinity set.")

def allocate_numa_tensor(num_elements, numa_node=0):
    """ NUMA 노드에 맞게 텐서를 할당 """
    tensor = torch.randn(num_elements, device=f"cpu:{numa_node}")
    print(f"Tensor allocated on NUMA node {numa_node}.")
    return tensor

def numa_tensor_operation(tensor):
    """ NUMA-aware 텐서 연산을 수행하는 예시 """
    # CPU 친화성 설정 (이 예시에서는 텐서를 CPU 0에 할당)
    set_numa_affinity(0)
    
    # 텐서 연산 (예: 두 텐서 더하기)
    result = tensor + tensor
    return result

def distribute_tensors_across_numa_nodes(num_elements, num_nodes=2):
    """ NUMA 노드 간에 텐서를 분배하고 연산을 수행 """
    tensors = []
    for i in range(num_nodes):
        tensor = allocate_numa_tensor(num_elements, numa_node=i)
        tensors.append(tensor)

    # 각 NUMA 노드에서 텐서 연산 수행
    results = []
    for tensor in tensors:
        result = numa_tensor_operation(tensor)
        results.append(result)

    # 최종 결과 합침
    final_result = sum(results)
    print(f"Final result after summing tensors: {final_result.sum().item()}")

def numa_test():
    num_elements = 1000000  # 1M 요소
    num_nodes = 2  # NUMA 노드 2개 사용

    # NUMA 노드 간 텐서 분배 및 연산 수행
    start_time = time.time()
    distribute_tensors_across_numa_nodes(num_elements, num_nodes)
    end_time = time.time()
    
    print(f"NUMA-aware computation completed in {end_time - start_time:.5f} seconds.")

# NUMA 최적화된 텐서 분배 및 연산 테스트 실행
if __name__ == "__main__":
    numa_test()
