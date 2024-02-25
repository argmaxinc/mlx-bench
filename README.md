# mlx-bench
Benchmark MLX Performance across forks and commits

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Example command:

```bash
python bench_mistral.py
    --repo-a ml-explore/mlx --commit-a 22364c40b7e488a81c322dfe5663a03daf3190a8
    --repo-b argmaxinc/mlx --commit-b 5bf706ad86cbe655f8efb1d5e84316e6efb35593
    --output-dir external --hub-model-name mlx-community/Mistral-7B-Instruct-v0.2-4-bit
```

This benchmark tests the two repo+commit pairs for correctness and plot the performance differential in `bench.png`. Useful for benchmarking pull requests such as [ml-explore/mlx#735](https://github.com/ml-explore/mlx/pull/735).
