## MBALL
Training and inference of large language models (LLMs) require massive system resources, often suffering from GPU memory shortages, system bottlenecks, and I/O delays that lead to high latency. To address these issues, we propose MBALL (Memory Ballooner), a novel system software framework that optimizes resource allocation across CPU, DRAM, GPU, and VRAM to drastically reduce model loading and inference delays. MBALL implements three core components: (1) a GPU Process Allocator to prevent GPU context collisions, (2) a Task Memory Manager for dynamic DRAM–VRAM memory redistribution, and (3) a GPU Booster with Swap Support to mitigate VRAM overflow via GPU-specific swap space. In experiments on real LLM workloads, MBALL reduced model loading time by up to 60%, significantly decreased VRAM usage, and delivered superior performance compared to existing approaches. The integrated architecture of MBALL combines versatility and practicality, making it applicable in both industry and academia. We describe MBALL’s design and components, and through comparative experiments with baseline systems, we demonstrate its effectiveness in mitigating latency issues in LLM environments.

## Environment Specification
- **CPU**: Intel Xeon Gold 6440 (40 cores)
- **GPU**: NVIDIA A100 with 80GB VRAM
- **DRAM**: 512 GB
- **NVMe SSD**: High-performance NVMe SSD (utilized as GPU-dedicated swap space)
- **Operating System**: Ubuntu 22.04 LTS
- **Container Environment**: Docker + OverlayFS lightweight runtime

## Reproducibility Limitations
- Experimental setup depends on **high-performance hardware** (large GPU VRAM, high-capacity DRAM, NVMe SSD).
- Quantitative results reported in the paper are **only fully reproducible** under equivalent hardware specifications.
- On smaller GPUs or without NVMe SSD support, performance gains may vary and could be reduced.

## Improvement Suggestions
- **Demo Script**:
  - Provide a lightweight version using smaller LLMs (e.g., LLaMA-7B, 13B) to validate MBALL’s functionality on lower-end GPUs.
  - Use reduced datasets and configurations to shorten runtime and improve accessibility.
- **Environment Setup Guide**:
  - Clearly specify mandatory vs. optional hardware requirements.
  - Include examples of alternative I/O configurations for environments without NVMe SSD.
- **Benchmark Mode**:
  - Add memory bottleneck simulation and synthetic load generation.
  - Allow users without high-end hardware to observe the algorithm’s behavior and effects.
