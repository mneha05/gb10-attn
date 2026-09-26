# ExecuTorch / XNNPACK probe

The export script lowers the same attention-score shape used by the native
microbenchmark to ExecuTorch's XNNPACK CPU backend.

```bash
python -m venv .venv-et
source .venv-et/bin/activate
pip install torch executorch
python arm_cpu/executorch/export_attention_probe.py   --tokens 2048 --dim 128 --out attention_xnnpack.pte
```

The output is an ExecuTorch `.pte` program suitable for the ExecuTorch runtime.
XNNPACK is the general CPU backend used by ExecuTorch on Arm64 Linux and
Android.

This path is intentionally separate from the handwritten Neon/SVE benchmark:
XNNPACK chooses its own optimized operators, while the native benchmark exposes
the exact vectorized inner loop for instruction-level analysis.
