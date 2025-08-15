import time
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

# 1. 시스템 기본 구조

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryTier(Enum):
    VRAM = "vram"
    DRAM = "dram"
    NVME = "nvme"

class ProcessState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    RUNNING = "running"
    SWAPPING = "swapping"

@dataclass
class MemoryStats:
    vram_used: int = 0
    vram_total: int = 0
    dram_used: int = 0
    dram_total: int = 0
    nvme_used: int = 0
    nvme_total: int = 0

    @property
    def vram_utilization(self) -> float:
        return (self.vram_used / self.vram_total) if self.vram_total > 0 else 0.0

    @property
    def dram_utilization(self) -> float:
        return (self.dram_used / self.dram_total) if self.dram_total > 0 else 0.0

@dataclass
class TaskConfig:
    task_id: str
    model_size: int        # bytes
    required_vram: int     # bytes
    required_dram: int     # bytes
    priority: int = 1
    max_swap_latency: float = 1000.0   # ms

@dataclass
class SwapOperation:
    source_tier: MemoryTier
    target_tier: MemoryTier
    data_size: int
    task_id: str
    timestamp: float

# 2. GPU Process Allocator (실제 GPU HW 의존 부분은 Mock)

class GPUProcessAllocator:
    def __init__(self, num_gpus: int = 2, max_processes_per_gpu: int = 4):
        self.num_gpus = num_gpus
        self.max_processes_per_gpu = max_processes_per_gpu
        self.gpu_processes: Dict[int, List[str]] = {i: [] for i in range(num_gpus)}
        self.process_gpu_mapping: Dict[str, int] = {}
        self.cuda_contexts: Dict[str, Any] = {}
        self.allocation_lock = threading.Lock()
        logger.info(f"GPUProcessAllocator initialized with {num_gpus} GPUs")

    def allocate_gpu(self, task_id: str, required_vram: int) -> Optional[int]:
        with self.allocation_lock:
            best_gpu = self._find_best_gpu(required_vram)
            if best_gpu is not None:
                self.gpu_processes[best_gpu].append(task_id)
                self.process_gpu_mapping[task_id] = best_gpu
                self._initialize_cuda_context(task_id, best_gpu)
                logger.info(f"Allocated GPU {best_gpu} to task {task_id}")
                return best_gpu
            logger.warning(f"Failed to allocate GPU for task {task_id}")
            return None

    def _find_best_gpu(self, required_vram: int) -> Optional[int]:
        gpu_scores = []
        for gpu_id in range(self.num_gpus):
            if len(self.gpu_processes[gpu_id]) >= self.max_processes_per_gpu:
                continue
            current_usage = len(self.gpu_processes[gpu_id]) * 2_000_000_000  # 2GB/process (Mock)
            total_vram = 80_000_000_000  # 80GB
            if current_usage + required_vram > total_vram:
                continue
            score = 1.0 - (current_usage / total_vram)
            gpu_scores.append((gpu_id, score))
        if gpu_scores:
            gpu_scores.sort(key=lambda x: x[1], reverse=True)
            return gpu_scores
        return None

    def _initialize_cuda_context(self, task_id: str, gpu_id: int):
        context_info = {
            'gpu_id': gpu_id,
            'initialized_at': time.time(),
            'context_handle': f"cuda_ctx_{task_id}_{gpu_id}"
        }
        self.cuda_contexts[task_id] = context_info
        logger.info(f"CUDA context initialized for task {task_id} on GPU {gpu_id}")

    def deallocate_gpu(self, task_id: str):
        with self.allocation_lock:
            if task_id in self.process_gpu_mapping:
                gpu_id = self.process_gpu_mapping[task_id]
                self.gpu_processes[gpu_id].remove(task_id)
                del self.process_gpu_mapping[task_id]
                if task_id in self.cuda_contexts:
                    del self.cuda_contexts[task_id]
                logger.info(f"Deallocated GPU {gpu_id} from task {task_id}")

    def get_gpu_utilization(self) -> Dict[int, float]:
        utilization = {}
        for gpu_id in range(self.num_gpus):
            process_count = len(self.gpu_processes[gpu_id])
            utilization[gpu_id] = process_count / self.max_processes_per_gpu
        return utilization

# 3. Task Memory Manager

class TaskMemoryManager:
    def __init__(self, vram_size: int = 80*1024**3, dram_size: int = 512*1024**3):
        self.vram_size = vram_size  # 80GB
        self.dram_size = dram_size  # 512GB
        self.vram_pool: Dict[str, int] = {}
        self.dram_pool: Dict[str, int] = {}
        self.nvme_pool: Dict[str, int] = {}
        self.vram_used = 0
        self.dram_used = 0
        self.nvme_used = 0
        self.swap_history: List[SwapOperation] = []
        self.memory_lock = threading.Lock()
        self.memory_blocks: Dict[str, List[Tuple[int, int]]] = {}
        logger.info("TaskMemoryManager initialized")

    def allocate_memory(self, task_id: str, size: int, preferred_tier: MemoryTier = MemoryTier.VRAM) -> bool:
        with self.memory_lock:
            if preferred_tier == MemoryTier.VRAM:
                return self._allocate_vram(task_id, size)
            elif preferred_tier == MemoryTier.DRAM:
                return self._allocate_dram(task_id, size)
            else:
                return self._allocate_nvme(task_id, size)

    def _allocate_vram(self, task_id: str, size: int) -> bool:
        if self.vram_used + size <= self.vram_size:
            self.vram_pool[task_id] = self.vram_pool.get(task_id, 0) + size
            self.vram_used += size
            self._track_memory_block(task_id, size, MemoryTier.VRAM)
            logger.info(f"Allocated {size} bytes VRAM to task {task_id}")
            return True
        else:
            logger.info(f"VRAM insufficient, trying DRAM for task {task_id}")
            return self._allocate_dram(task_id, size)

    def _allocate_dram(self, task_id: str, size: int) -> bool:
        if self.dram_used + size <= self.dram_size:
            self.dram_pool[task_id] = self.dram_pool.get(task_id, 0) + size
            self.dram_used += size
            self._track_memory_block(task_id, size, MemoryTier.DRAM)
            logger.info(f"Allocated {size} bytes DRAM to task {task_id}")
            return True
        else:
            logger.info(f"DRAM insufficient, trying NVMe for task {task_id}")
            return self._allocate_nvme(task_id, size)

    def _allocate_nvme(self, task_id: str, size: int) -> bool:
        self.nvme_pool[task_id] = self.nvme_pool.get(task_id, 0) + size
        self.nvme_used += size
        self._track_memory_block(task_id, size, MemoryTier.NVME)
        logger.info(f"Allocated {size} bytes NVMe to task {task_id}")
        return True

    def _track_memory_block(self, task_id: str, size: int, tier: MemoryTier):
        if task_id not in self.memory_blocks:
            self.memory_blocks[task_id] = []
        start_addr = len(self.memory_blocks[task_id]) * 1024 + hash(tier.value) % 1000000
        self.memory_blocks[task_id].append((start_addr, size))

    def swap_memory(self, task_id: str, source_tier: MemoryTier, target_tier: MemoryTier, size: int) -> float:
        start_time = time.time()
        with self.memory_lock:
            swap_latency = self._calculate_swap_latency(source_tier, target_tier, size)
            self._update_memory_pools(task_id, source_tier, target_tier, size)
            swap_op = SwapOperation(
                source_tier=source_tier,
                target_tier=target_tier,
                data_size=size,
                task_id=task_id,
                timestamp=time.time()
            )
            self.swap_history.append(swap_op)
            time.sleep(swap_latency / 1000.0)
            end_time = time.time()
            actual_latency = (end_time - start_time) * 1000
            logger.info(f"Swapped {size} bytes from {source_tier.value} to {target_tier.value} for task {task_id}, latency: {actual_latency:.2f}ms")
            return actual_latency

    def _calculate_swap_latency(self, source: MemoryTier, target: MemoryTier, size: int) -> float:
        rates = {
            (MemoryTier.VRAM, MemoryTier.DRAM): 600000,
            (MemoryTier.DRAM, MemoryTier.NVME): 7000,
            (MemoryTier.VRAM, MemoryTier.NVME): 6000
        }
        rate = rates.get((source, target), 1000)
        size_mb = size / (1024 ** 2)
        return size_mb / rate * 1000

    def _update_memory_pools(self, task_id: str, source: MemoryTier, target: MemoryTier, size: int):
        if source == MemoryTier.VRAM:
            self.vram_pool[task_id] = max(0, self.vram_pool.get(task_id, 0) - size)
            self.vram_used = max(0, self.vram_used - size)
        elif source == MemoryTier.DRAM:
            self.dram_pool[task_id] = max(0, self.dram_pool.get(task_id, 0) - size)
            self.dram_used = max(0, self.dram_used - size)
        if target == MemoryTier.VRAM:
            self.vram_pool[task_id] = self.vram_pool.get(task_id, 0) + size
            self.vram_used += size
        elif target == MemoryTier.DRAM:
            self.dram_pool[task_id] = self.dram_pool.get(task_id, 0) + size
            self.dram_used += size

    def get_memory_stats(self) -> MemoryStats:
        return MemoryStats(
            vram_used=self.vram_used,
            vram_total=self.vram_size,
            dram_used=self.dram_used,
            dram_total=self.dram_size
        )


