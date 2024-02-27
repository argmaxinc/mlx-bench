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
    --repo-a ml-explore/mlx --commit-a 22364c40b7e488a81c322dfe5663a03daf3190a8 \
    --repo-b argmaxinc/mlx --commit-b e44480a03d53d9f0895b9f8156ad8c045eb10d4e \
    --output-dir external --hub-model-name mlx-community/Mistral-7B-Instruct-v0.2-4-bit
```

This benchmark tests the two repo+commit pairs for correctness and plot the performance differential in `bench.png`.
