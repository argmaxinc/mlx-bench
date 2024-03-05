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
    --repo-a ml-explore/mlx --commit-a 7b463ffb077076e239b8931349d54fd5832b248c \
    --repo-b ml-explore/mlx --commit-b 0787724c44b870943386fe97ff709ab535f62c9c \
    --output-dir external --hub-model-name mlx-community/Mistral-7B-Instruct-v0.2-4-bit \
    --max-context-length 800 \
    --fail-for-mismatch-before-n-tokens 800
```

This benchmark tests the two repo+commit pairs for correctness and plot the performance. These are saved in the output directory.
