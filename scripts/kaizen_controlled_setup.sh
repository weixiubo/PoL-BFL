#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
circom_bin=${CIRCOM_BIN:-circom}
node_bin=${NODE_BIN:-node}
snarkjs_cli=${SNARKJS_CLI:-"$repo_root/node_modules/snarkjs/cli.js"}
export NODE_OPTIONS="${NODE_OPTIONS:-} --max-old-space-size=32768"
ptau=${1:?usage: kaizen_controlled_setup.sh /path/to/power22.ptau [build-dir]}
build=${2:-"$repo_root/circuits/final/build/kaizen-controlled"}
config="$repo_root/config/kaizen_controlled.json"
circuit="$repo_root/circuits/final/kaizen_controlled_cost.circom"
expected_blake2b=0d64f63dba1a6f11139df765cb690da69d9b2f469a1ddd0de5e4aa628abb28f787f04c6a5fb84a235ec5ea7f41d0548746653ecab0559add658a83502d1cb21b

[[ -f "$ptau" && -f "$config" && -f "$circuit" ]] || {
  echo "controlled baseline input is missing" >&2
  exit 1
}
observed_blake2b=$(b2sum "$ptau" | awk '{print $1}')
[[ "$observed_blake2b" == "$expected_blake2b" ]] || {
  echo "power-22 BLAKE2b-512 mismatch" >&2
  exit 1
}
mkdir -p "$build"
r1cs="$build/kaizen_controlled_cost.r1cs"
wasm="$build/kaizen_controlled_cost_js/kaizen_controlled_cost.wasm"
cpp="$build/kaizen_controlled_cost_cpp/kaizen_controlled_cost.cpp"
if [[ ${REUSE_COMPILED:-0} == 1 ]]; then
  for compiled in "$r1cs" "$wasm" "$cpp"; do
    [[ -s "$compiled" ]] || {
      echo "controlled compiled artifact is unavailable for resume: $compiled" >&2
      exit 1
    }
  done
  echo "Reusing complete controlled circuit compilation"
else
  "$circom_bin" "$circuit" --r1cs --wasm --c --sym -o "$build"
fi
r1cs_info="$build/r1cs-info.log"
"$node_bin" "$snarkjs_cli" r1cs info "$r1cs" | tee "$r1cs_info"
constraints=$(sed -n 's/.*# of Constraints: *\([0-9][0-9]*\).*/\1/p' "$r1cs_info" | tail -n 1)
[[ -n "$constraints" && "$constraints" -ge 3800000 && "$constraints" -le 4200000 ]] || {
  echo "controlled baseline constraint count is outside the declared range: $constraints" >&2
  exit 1
}
make -C "$build/kaizen_controlled_cost_cpp" -j "$(nproc)"
zkey0="$build/kaizen_controlled_cost_0000.zkey"
zkey1="$build/kaizen_controlled_cost_0001.zkey"
zkey="$build/kaizen_controlled_cost_final.zkey"
if [[ ${REUSE_ZKEY0:-0} == 1 && -s "$zkey0" ]]; then
  echo "Reusing complete initial circuit zkey"
else
  "$node_bin" "$snarkjs_cli" groth16 setup "$r1cs" "$ptau" "$zkey0"
fi
openssl rand -hex 64 | "$node_bin" "$snarkjs_cli" zkey contribute \
  "$zkey0" "$zkey1" --name="PoL-BFL controlled baseline phase 2"
beacon=$(openssl rand -hex 32)
"$node_bin" "$snarkjs_cli" zkey beacon "$zkey1" "$zkey" \
  "$beacon" 10 -n="PoL-BFL controlled baseline beacon"
unset beacon
"$node_bin" "$snarkjs_cli" zkey verify "$r1cs" "$ptau" "$zkey" \
  | tee "$build/zkey-verify.log"
"$node_bin" "$snarkjs_cli" zkey export verificationkey "$zkey" \
  "$build/verification_key.json"
cp "$config" "$build/config.json"
printf '{"seed":"2"}\n' > "$build/input.json"
"$build/kaizen_controlled_cost_cpp/kaizen_controlled_cost" \
  "$build/input.json" "$build/benchmark.wtns"
python_bin=${PYTHON_BIN:-python3}
"$python_bin" - "$build" "$observed_blake2b" "$constraints" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
record = {
    "schema_version": 1,
    "classification": "controlled_kaizen_style_cost_baseline",
    "powers_of_tau_blake2b_512": sys.argv[2],
    "constraints": int(sys.argv[3]),
    "artifacts": {},
}
for name in (
    "kaizen_controlled_cost.r1cs",
    "kaizen_controlled_cost_final.zkey",
    "verification_key.json",
    "kaizen_controlled_cost_js/kaizen_controlled_cost.wasm",
    "kaizen_controlled_cost_cpp/kaizen_controlled_cost",
    "benchmark.wtns",
    "zkey-verify.log",
):
    path = root / name
    record["artifacts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
body = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
record["record_digest"] = hashlib.sha256(body).hexdigest()
(root / "controlled_setup.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
sha256sum "$build/controlled_setup.json" "$r1cs" "$zkey"
