# NVIDIA Kepler GPU Support (sm_35 / sm_37) in llama.cpp

This branch adds and maintains support for **NVIDIA Kepler GPUs** (Compute Capability 3.5 and 3.7) in modern `llama.cpp`.

Supported GPUs include:
* **sm_35**: GeForce GT 730 (GK208B), GeForce GTX 780 / 780 Ti, Tesla K40, Tesla K20
* **sm_37**: Tesla K80 (GK210)

Kepler support was dropped upstream from CUDA 12.0+ and modern `llama.cpp` architecture presets. This fork re-enables targeting these legacy architectures when building with CUDA 11.x.

---

## Prerequisites

1. **NVIDIA Driver**: 470.xx (Kepler legacy series)
2. **CUDA Toolkit**: 11.x (e.g. CUDA 11.8) — Kepler is removed in CUDA 12+
3. **Host Compiler**: GCC <= 11 (e.g. GCC 11.4)
4. **CMake**: >= 3.18

---

## Build Instructions

```bash
mkdir -p build && cd build

cmake .. \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=35 \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF

# On systems with legacy CPUs lacking AVX/BMI2 (e.g. Core 2 Duo):
# add: -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_SSE42=OFF -DGGML_BMI2=OFF

cmake --build . --target llama-cli llama-server -j$(nproc)
```

> **Note on Required Flags**:
> * `-DCMAKE_CUDA_ARCHITECTURES=35` (or `37` for Tesla K80): targets the Kepler compute capability.
> * `-DGGML_CUDA_GRAPHS=OFF`: Required; CUDA Graphs require sm_60+ and will crash on Kepler.
> * `-DGGML_NATIVE=OFF`: Prevents host CPU detection from overriding GPU targets.

---

## Device Verification

Verify that your GPU is detected:
```bash
./bin/llama-cli --list-devices
```
Output:
```text
Available devices:
  CUDA0: NVIDIA GeForce GT 730 (2000 MiB, 1495 MiB free)
```
