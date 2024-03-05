# mlx-bench
Benchmark MLX Performance across forks and commits

## Installation

```bash
pip install -r requirements.txt
git submodule init && git submodule update
```

## Usage

Example command to test [ml-explore/mlx#735](https://github.com/ml-explore/mlx/pull/735) for Mistral 7b:

```bash
python bench_mistral.py \
    --repo-a ml-explore/mlx --commit-a c096a77b9b012a4cc91edcf50683c495a421dcb7 \
    --repo-b argmaxinc/mlx --commit-b cd6e306b52f2c80eee9662bb225b853796aa48e5 \
    --output-dir external --hub-model-name mlx-community/Mistral-7B-Instruct-v0.2-4-bit \
    --max-context-length 4100 \
    --fail-for-mismatch-before-n-tokens 4100
```

This benchmark tests the two repo+commit pairs for correctness and plot the performance. These are saved in the output directory.
