import torch
import os
import numpy as np
import time

def set_numa_affinity(cpu_id):
    """ 주어진 CPU ID에 NUMA 노드 할당 """
    os.sched_setaffinity(0, {cpu_id})  # CPU 친화성 설정
    print(f"CPU {cpu_id} affinity set.")

def allocate_numa_memory(num_elements, numa_node=0):
    """ NUMA 노드에 맞게 메모리 할당 """
    # CPU와 NUMA 노드에 맞는 메모리 할당 (PyTorch 텐서 사용)
    tensor = torch.randn(num_elements, device=f"cpu:{numa_node}")
    print(f"Memory allocated on NUMA node {numa_node}.")
    return tensor

def run_numa_computation(tensor):
    """ NUMA-aware 연산을 수행하는 예시 """
    # CPU 친화성 설정 (이 예시에서는 텐서를 CPU 0에 할당)
    set_numa_affinity(0)
    
    # 텐서 연산 (예: 두 텐서 더하기)
    result = tensor + tensor
    return result

# NUMA 환경에서 메모리 할당 및 계산 예시
def numa_test():
    num_elements = 1000000  # 1M 요소
    numa_node = 0  # NUMA 노드 0에 메모리 할당
    
    # 메모리 할당
    tensor = allocate_numa_memory(num_elements, numa_node)
    
    # NUMA-aware 연산 수행
    start_time = time.time()
    result = run_numa_computation(tensor)
    end_time = time.time()
    
    print(f"Computation completed in {end_time - start_time:.5f} seconds.")

# NUMA 최적화된 메모리 관리 테스트 실행
if __name__ == "__main__":
    numa_test()
