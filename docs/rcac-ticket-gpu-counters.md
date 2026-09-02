# RCAC support ticket — GPU performance counter access on the `gb10` partition

Copy the block below into a ticket at https://www.rcac.purdue.edu/help
(or email rcac-help@purdue.edu). Subject line first.

---

**Subject:** Request: enable GPU performance counters for non-root users on rowdy `gb10` partition (ERR_NVGPUCTRPERM)

Hello,

I am running CUDA kernel performance work on the `gb10` partition on rowdy
(account `mithuna`, user `mahesh54`) and I am blocked from profiling by a
driver-level permission setting.

**What I am trying to do**

Profile my own CUDA kernels with NVIDIA Nsight Compute to collect memory
throughput and occupancy counters, so I can measure what fraction of the GB10's
memory bandwidth my kernels achieve. This is read-only profiling of my own
process; it does not require elevated privileges for anything else.

**What happens**

Running `ncu` on a `gb10` compute node (c001) fails with:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
GPU Performance Counters on the target device 0. For instructions on enabling
permissions and to get more information see
https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

Nsight Compute connects to the device and launches the application correctly —
only the counter collection is refused, so this is the driver's profiling
restriction rather than a tool or architecture compatibility problem.

**Reproducer**

```bash
sinteractive -A mithuna -p gb10 -c 20 --gres=gpu:1 -t 00:10:00
/apps/cuda/12.8/bin/ncu --metrics dram__bytes_read.sum python -c "
import torch; x=torch.randn(2**24, device='cuda'); (x*2).sum().item()"
```

**What would fix it**

By default the NVIDIA driver restricts performance counters to root
(`NVreg_RestrictProfilingToAdminUsers=1`). The documented fix is to set it to 0
in the kernel module options on the GPU nodes, e.g. a file such as
`/etc/modprobe.d/nvidia-profiling.conf` containing:

```
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

followed by a reboot or module reload. NVIDIA's guidance is here:
https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters

If a global change is not acceptable, I would be glad to take any narrower
alternative — enabling it only on the `gb10` nodes, granting my account access,
or a reserved node/time window where profiling is permitted.

**Notes on impact**

I understand this setting exists because performance counters can expose
information across GPU contexts. On the `gb10` nodes, jobs receive an exclusive
whole-GPU allocation (`--gres=gpu:1` on a single-GPU node), so a user profiling
their own job would not be sharing the device with another user's work. If that
reasoning is wrong for this cluster, please disregard it — I am happy to work
within whatever constraint you prefer.

**Environment**

- Cluster / partition: rowdy, `gb10` (nodes c000–c004; observed on c001)
- GPU: NVIDIA GB10, compute capability 12.1, driver 595.58.03
- Architecture: aarch64
- Nsight Compute: 2025.1.1.0, from `/apps/cuda/12.8`
- CUDA used to build: 13.2 (`/usr/local/cuda-13.2`) — note the Lmod `cuda/12.8`
  module's `nvcc` cannot target `sm_121` at all, which may be worth knowing for
  other `gb10` users; `--list-gpu-code` there stops at `sm_120`.

Thank you,
Neha Mahesh (mahesh54)

---

## Secondary item, optional to include

The `cuda/12.4`, `cuda/12.5`, `cuda/12.8` Lmod modules on rowdy predate GB10 and
cannot generate `sm_121` code. Users who `module load cuda` on a `gb10` node and
build a CUDA extension will silently get `sm_120` binaries (which run, via
minor-version compatibility, so the mis-targeting is invisible). A `cuda/13.x`
module pointing at the existing `/usr/local/cuda-13.2` install would prevent
that. Worth raising as a separate low-priority ticket rather than bundling it
with the profiling request.
