#!/usr/bin/env bash
set -euo pipefail

REF="${KLEIDIAI_REF:-v1.25.0}"
ROOT="${1:-$(pwd)/third_party/kleidiai}"
BUILD="${2:-$(pwd)/build-kleidiai}"

if [[ ! -d "$ROOT/.git" ]]; then
  git clone --depth 1 --branch "$REF"     https://github.com/ARM-software/kleidiai.git "$ROOT"
fi

cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"${JOBS:-4}"

echo "KleidiAI $REF built in $BUILD"
