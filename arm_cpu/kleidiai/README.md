# KleidiAI integration lane

This directory keeps the Arm ML-microkernel work separate from the handwritten
attention dot-product benchmark.

KleidiAI is Arm's low-level ML microkernel library. It exposes architecture-
specialized variants across Advanced SIMD/Neon, SVE and SME/SME2 rather than
providing a full model runtime.

## Build the official library

```bash
cd arm_cpu
./kleidiai/build_kleidiai.sh
```

The script pins the default integration target to **v1.25.0** unless
`KLEIDIAI_REF` is set.

## Why it belongs next to this benchmark

The handwritten path answers a narrow question precisely:

> what does the Q·K inner product cost on the GB10 Arm CPU when written as
> scalar, Neon, and SVE code?

KleidiAI answers the larger inference-runtime question by providing packing and
matrix microkernels specialized for Arm instruction features such as DotProd,
I8MM, SVE, SME and SME2.

A useful next comparison on hardware is:

1. handwritten FP32 attention score kernel,
2. XNNPACK/ExecuTorch CPU delegation,
3. a KleidiAI matmul microkernel matched to the detected feature set.

Do not select an SME/SME2 microkernel from its name alone. The runtime probe in
`src/features.cpp` reports Linux `HWCAP2_SME`/`HWCAP2_SME2`, and KleidiAI
documents additional ABI/compiler requirements for SME call chains.
