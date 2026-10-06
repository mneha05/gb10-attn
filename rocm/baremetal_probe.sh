#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-rocm-baremetal-provenance.txt}"
{
  echo "=== timestamp ==="; date -u +%FT%TZ
  echo "=== uname ==="; uname -a
  echo "=== cpu ==="; lscpu || true
  echo "=== numa ==="; numactl --hardware 2>/dev/null || true
  echo "=== pci accelerators ==="; lspci -nn | grep -Ei "vga|display|3d|amd|advanced micro devices" || true
  echo "=== kernel cmdline ==="; cat /proc/cmdline || true
  echo "=== hugepages ==="; grep -i huge /proc/meminfo || true
  echo "=== iommu ==="; dmesg 2>/dev/null | grep -i iommu | tail -40 || true
  echo "=== rocm version ==="; cat /opt/rocm/.info/version 2>/dev/null || true
  echo "=== hipconfig ==="; hipconfig --full 2>/dev/null || true
  echo "=== rocminfo ==="; rocminfo 2>/dev/null || true
  echo "=== rocm-smi ==="; rocm-smi 2>/dev/null || true
} | tee "$OUT"
