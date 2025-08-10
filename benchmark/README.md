우분투에서 AI모델 성능을 평가하기 위해 사용할 수 있는 벤치마킹 툴들입니다.
아래는 **GPU 성능**, **메모리 사용량**, **CPU 성능** 및 **전체 시스템 성능**을 측정할 수 있는 주요 툴들을 소개합니다:

### 1. **NVIDIA-smi (NVIDIA System Management Interface)**

* **목적**: GPU 성능 모니터링 및 벤치마킹
* **설명**: `nvidia-smi`는 NVIDIA GPU의 성능, 메모리 사용량, 온도 및 기타 상태 정보를 제공합니다. 주로 GPU 상태를 모니터링하거나 벤치마킹 할 때 유용합니다.
* **주요 기능**:

  * GPU 메모리 사용량
  * GPU 온도
  * GPU 사용률 및 성능 모니터링
* **사용법**:

  ```bash
  nvidia-smi
  ```
* **추가 툴**: `nvidia-smi`는 `nvidia-smi -l`로 실시간으로 모니터링을 수행할 수 있습니다.

### 2. **Benchmarking with `perf`**

* **목적**: 시스템 성능 (CPU, 메모리 등) 측정
* **설명**: `perf`는 리눅스에서 제공하는 성능 분석 도구로, CPU 성능, 메모리 사용량, 캐시 성능 등을 분석할 수 있습니다. 벤치마킹을 통해 애플리케이션의 성능을 측정할 수 있습니다.
* **주요 기능**:

  * CPU 사용률 분석
  * 메모리 접근 및 캐시 분석
  * 이벤트 기반 프로파일링
* **사용법**:

  ```bash
  sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses sleep 10
  ```

### 3. **`fio` (Flexible I/O Tester)**

* **목적**: 디스크 성능 벤치마킹
* **설명**: `fio`는 디스크 I/O 성능을 벤치마킹하는 도구로, 다양한 입출력 작업을 생성하여 디스크 성능을 테스트할 수 있습니다. 이 도구는 스토리지 성능을 벤치마킹하고 성능을 측정하는 데 유용합니다.
* **주요 기능**:

  * 디스크 읽기/쓰기 속도 측정
  * 임의/순차 읽기/쓰기 벤치마킹
  * 다양한 I/O 스케줄러 테스트
* **사용법**:

  ```bash
  sudo apt install fio
  fio --name=mytest --ioengine=sync --rw=write --bs=4k --numjobs=8 --size=1G --runtime=60 --time_based --output-format=json
  ```

### 4. **`stress`**

* **목적**: CPU, 메모리 및 I/O 부하 생성
* **설명**: `stress`는 시스템의 부하를 시뮬레이션하는 도구로, CPU, 메모리, I/O 등을 부하를 주어 시스템 성능을 테스트할 수 있습니다.
* **주요 기능**:

  * CPU, 메모리 및 I/O 부하 테스트
  * 시스템 안정성 테스트
* **사용법**:

  ```bash
  sudo apt install stress
  stress --cpu 4 --io 2 --vm 2 --vm-bytes 128M --timeout 60s
  ```

### 5. **`htop`**

* **목적**: 시스템 성능 모니터링
* **설명**: `htop`은 시스템 리소스(CPU, 메모리, 스왑 등)의 실시간 모니터링 툴로, 우분투에서 가장 많이 사용됩니다. 다양한 프로세스와 리소스 사용 현황을 실시간으로 확인할 수 있습니다.
* **주요 기능**:

  * CPU, 메모리, 스왑 사용량 실시간 모니터링
  * 프로세스 관리
  * 자원 사용량에 따른 프로세스 우선순위 변경
* **사용법**:

  ```bash
  sudo apt install htop
  htop
  ```

### 6. **`py-spy`**

* **목적**: Python 애플리케이션의 CPU 및 메모리 성능 모니터링
* **설명**: `py-spy`는 Python 애플리케이션의 성능을 실시간으로 프로파일링하는 도구입니다. CPU와 메모리 사용을 시각적으로 분석할 수 있습니다.
* **주요 기능**:

  * Python 애플리케이션의 실시간 프로파일링
  * CPU 및 메모리 사용 분석
* **사용법**:

  ```bash
  sudo apt install py-spy
  py-spy top --pid <your-python-process-id>
  ```

### 7. **`nvprof` (NVIDIA Profiler)**

* **목적**: NVIDIA GPU 성능 분석
* **설명**: `nvprof`는 GPU의 성능을 프로파일링하는 NVIDIA의 도구로, CUDA 기반 애플리케이션의 성능을 분석하는 데 유용합니다.
* **주요 기능**:

  * GPU 실행의 성능 분석
  * 커널 실행 시간 및 메모리 사용 추적
* **사용법**:

  ```bash
  nvprof python your_script.py
  ```

### 8. **`cudnn`**

* **목적**: GPU 연산 성능 최적화
* **설명**: NVIDIA cuDNN은 딥러닝 연산 최적화 라이브러리로, 딥러닝 모델의 훈련과 추론을 최적화하는 데 사용됩니다. cuDNN을 통해 GPU의 성능을 최대한 활용할 수 있습니다.
* **사용법**:
  `cudnn`은 NVIDIA GPU 환경에서 자동으로 활성화되며, CUDA 기반 딥러닝 프레임워크에서 성능 최적화가 이루어집니다.

### 9. **`dstat`**

* **목적**: 시스템 자원 사용 모니터링
* **설명**: `dstat`는 시스템의 다양한 자원(CPU, 메모리, 네트워크, 디스크 등)을 실시간으로 모니터링하는 도구입니다. 
성능 분석과 리소스 모니터링에 유용합니다.
* **주요 기능**:

  * CPU, 메모리, 네트워크, 디스크 사용 모니터링
  * 성능 측정을 위한 여러 지표 제공
* **사용법**:

  ```bash
  sudo apt install dstat
  dstat
  ```

### 10. **`sysbench`**

* **목적**: CPU, 메모리 및 I/O 성능 벤치마킹
* **설명**: `sysbench`는 시스템 성능을 테스트하는 벤치마킹 툴로, CPU, 메모리 및 디스크 I/O 성능을 측정할 수 있습니다.
* **주요 기능**:

  * CPU 성능 벤치마킹
  * 메모리 성능 및 I/O 성능 측정
* **사용법**:

  ```bash
  sudo apt install sysbench
  sysbench cpu --cpu-max-prime=20000 run
  ```

### 결론

* **GPU 성능**: `nvidia-smi`, `nvprof`, `py-spy` (GPU 성능 모니터링 및 분석)
* **CPU 및 메모리 성능**: `perf`, `htop`, `stress`, `sysbench`
* **I/O 성능**: `fio`, `dstat`
* **전체 시스템 성능**: `stress`, `dstat`, `sysbench`

이 도구들을 통해 우분투에서 시스템 리소스를 모니터링하고 성능을 벤치마킹할 수 있습니다. 각 도구는 특정 성능 요소를 최적화하는 데 유용하며, 여러 툴을 조합하여 시스템의 전반적인 성능을 평가할 수 있습니다.
