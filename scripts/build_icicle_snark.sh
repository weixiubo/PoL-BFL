#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_root=${POLBFL_ICICLE_SOURCE_ROOT:-"$repo_root/.tools/icicle-snark/source"}
install_root=${POLBFL_ICICLE_INSTALL_ROOT:-"$repo_root/.tools/icicle-snark"}
cuda_root=${POLBFL_CUDA_ROOT:-/usr/local/cuda-12.4}
rust_cargo=${POLBFL_CARGO_BIN:-"$HOME/.cargo/bin/cargo"}
upstream_commit=bf00385db19087a8a3f6d754b43b006254b1b465
memory_patch="$repo_root/tools/icicle_snark/0001-limit-msm-memory.patch"

if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
  echo "missing CUDA compiler: $cuda_root/bin/nvcc" >&2
  exit 1
fi
if [[ ! -x "$rust_cargo" ]]; then
  echo "missing Cargo executable: $rust_cargo" >&2
  exit 1
fi
if [[ ! -d "$source_root/.git" ]]; then
  mkdir -p "$(dirname "$source_root")"
  git clone https://github.com/ingonyama-zk/icicle-snark.git "$source_root"
fi
if [[ -n "$(git -C "$source_root" status --porcelain --untracked-files=no)" ]]; then
  echo "ICICLE-Snark source checkout has tracked modifications" >&2
  exit 1
fi
git -C "$source_root" fetch --quiet origin "$upstream_commit"
git -C "$source_root" checkout --quiet --detach "$upstream_commit"
cp "$repo_root/tools/icicle_snark/Cargo.lock" "$source_root/Cargo.lock"
if [[ ! -f "$memory_patch" ]]; then
  echo "missing locked ICICLE-Snark memory patch: $memory_patch" >&2
  exit 1
fi
git -C "$source_root" apply --unidiff-zero --check "$memory_patch"
git -C "$source_root" apply --unidiff-zero "$memory_patch"
cleanup_source() {
  git -C "$source_root" apply -R --unidiff-zero --check "$memory_patch" >/dev/null 2>&1 || return 0
  git -C "$source_root" apply -R --unidiff-zero "$memory_patch"
}
trap cleanup_source EXIT


PATH="$cuda_root/bin:$PATH" cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCURVE=bn254 \
  -DCUDA_BACKEND=local \
  -S "$source_root/icicle" \
  -B "$source_root/icicle/build"
PATH="$cuda_root/bin:$PATH" cmake --build "$source_root/icicle/build" -j "$(nproc)"
ICICLE_BACKEND_INSTALL_DIR="$source_root/icicle/build/backend" \
  "$rust_cargo" build \
  --release \
  --locked \
  --manifest-path "$source_root/Cargo.toml"

install -d -m 0755 \
  "$install_root/bin" \
  "$install_root/lib" \
  "$install_root/backend/cuda"
install -m 0755 "$source_root/target/release/icicle-snark" "$install_root/bin/icicle-snark"
install -m 0755 \
  "$source_root/icicle/build/libicicle_device.so" \
  "$source_root/icicle/build/libicicle_field_bn254.so" \
  "$source_root/icicle/build/libicicle_curve_bn254.so" \
  "$install_root/lib/"
install -m 0755 "$source_root"/icicle/build/backend/cuda/*.so "$install_root/backend/cuda/"

sha256sum \
  "$install_root/bin/icicle-snark" \
  "$install_root/lib"/*.so \
  "$install_root/backend/cuda"/*.so
