#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VEC="${1:-tests/vectors/synthetic.txt}"
mkdir -p work results
python -m src.quantization.export_vhdl_params
python -m src.quantization.gen_vectors
ghdl -a --std=08 --workdir=work rtl/units/mlp_pkg.vhd rtl/top/eye_mlp_params.vhd rtl/layers/dense_layer.vhd rtl/top/eye_mlp_top.vhd rtl/tb/tb_eye_mlp.vhd
ghdl -e --std=08 --workdir=work tb_eye_mlp
ghdl -r --std=08 --workdir=work tb_eye_mlp -gVEC_FILE="$VEC"
