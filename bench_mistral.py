import os
import json
import subprocess
from argmaxtools import utils, test_utils
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt

logger = utils.get_logger(__name__)

SETUP_CMD = "env CMAKE_BUILD_PARALLEL_LEVEL="" pip install -e ."
MAX_CONTEXT = 10000
MEASURE_EVERY_N_TOKENS = 100
BENCH_CMD = "python llms/mistral/mistral.py"
" --model-path ../external/model "
f" --max-tokens {MAX_CONTEXT} "
f" --tokens-per-eval {MEASURE_EVERY_N_TOKENS}"


def setup_mlx_repos(args):
    # Setup repo A
    repo_owner_a, repo_name_a = args.repo_a.rsplit("/")
    logger.info(f"Cloning repo A: {repo_owner_a}/{repo_name_a}@{args.commit_a}")
    utils._maybe_git_clone(
        out_dir=os.path.join(args.output_dir, "repo_a"),
        hub_url=args.hub_url,
        repo_name=repo_name_a,
        repo_owner=repo_owner_a,
        commit_hash=args.commit_a,
    )

    # Setup repo B
    repo_owner_b, repo_name_b = args.repo_b.rsplit("/")
    logger.info(f"Cloning repo B: {repo_owner_b}/{repo_name_b}@{args.commit_b}")
    utils._maybe_git_clone(
        out_dir=os.path.join(args.output_dir, "repo_b"),
        hub_url=args.hub_url,
        repo_name=repo_name_b,
        repo_owner=repo_owner_b,
        commit_hash=args.commit_b,
    )


def download_model(args):
    # Download the hub model
    snapshot_download(
        repo_id=args.hub_model_name,
        local_dir=os.path.join(args.output_dir, "model"),
        local_dir_use_symlinks=True
    )


def bench(args):
    bench_data = {}

    # # Install repo A
    # logger.info("Installing repo A")
    # subprocess.check_call(SETUP_CMD, shell=True, cwd=os.path.join(args.output_dir, "repo_a", args.repo_a.split("/")[1]))

    # Benchmarking --bench-cmd under repo A
    logger.info("Benchmarking repo A")
    subprocess.check_call(
        args.bench_cmd + " --benchmark-json-path benchmark_a.json " + args.bench_cmd_extra_args_a,
        shell=True, cwd=os.path.join(os.getcwd(), "mlx-examples"))

    # Load benchmark data from repo A
    with open(os.path.join(os.getcwd(), "mlx-examples", "benchmark_a.json")) as f:
        bench_data["repo_a"] = json.load(f)

    # # Install repo B
    # logger.info("Installing repo B")
    # subprocess.check_call(SETUP_CMD, shell=True, cwd=os.path.join(args.output_dir, "repo_b", args.repo_a.split("/")[1]))

    # Benchmarking --bench-cmd under repo B
    logger.info("Benchmarking repo B")
    subprocess.check_call(
        args.bench_cmd + " --benchmark-json-path benchmark_b.json --optimized-sdpa",
        shell=True, cwd=os.path.join(os.getcwd(), "mlx-examples"))

    # Load benchmark data from repo B
    with open(os.path.join(os.getcwd(), "mlx-examples", "benchmark_b.json")) as f:
        bench_data["repo_b"] = json.load(f)

    return bench_data


def check_correctness(bench_data):
    # Check correctness
    for result_a, result_b in zip(bench_data["repo_a"], bench_data["repo_b"]):
        if not result_a[-1] == result_b[-1]:
            logger.error(f"Mismatch in results: {result_a[-1]} vs {result_b[-1]}")
        else:
            logger.info(f"Results match: {result_a[-1]} vs {result_b[-1]}")


def plot_performance(bench_data, inference_ctx, args):
    # Plot performance
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in bench_data["repo_a"]], [x[1] for x in bench_data["repo_a"]], label=f"{args.repo_a}@{args.commit_a[:7]}")
    ax.plot([x[0] for x in bench_data["repo_b"]], [x[1] for x in bench_data["repo_b"]], label=f"{args.repo_b}@{args.commit_b[:7]}")
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Inference Speed (tokens/second)")
    ax.set_title(
        f"{inference_ctx['device_spec']['product_name']} "
        f"({inference_ctx['device_spec']['gpu_core_count']} GPU cores, "
        f"macOS={inference_ctx['os_spec']['os_build_number']})"
    )
    ax.legend()

    plt.savefig("bench.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the benchmark')
    parser.add_argument("--hub-url", type=str, help="URL to the code hub", default="github.com")
    parser.add_argument("--repo-a", type=str, help="Path to repo A (Syntax: owner/repo, e.g. ml-explore/mlx)")
    parser.add_argument("--repo-b", type=str, help="Path to repo B (Syntax: owner/repo), e.g. argmaxinc/mlx")
    parser.add_argument("--commit-a", type=str, help="Commit hash for repo A")
    parser.add_argument("--commit-b", type=str, help="Commit hash for repo B")
    parser.add_argument("--output-dir", type=str, help="Output directory for the benchmark")
    parser.add_argument(
        "--hub-model-name",
        type=str,
        choices=("mlx-community/Mistral-7B-Instruct-v0.2-4-bit",)
    )

    args = parser.parse_args()

    class BenchContext(test_utils.AppleSiliconContextMixin, test_utils.InferenceContextSpec):
        def code_spec(self):
            return {"repo_a": args.repo_a, "repo_b": args.repo_b,
                    "commit_a": args.commit_a, "commit_b": args.commit_b}

        def model_spec(self):
            return {"hub_model_name": args.hub_model_name}

    inference_ctx = BenchContext().spec_dict()

    logger.info("Running the benchmark with the following context:")
    from pprint import pprint
    pprint(inference_ctx)

    setup_mlx_repos(args)
    download_model(args)

    bench_data = bench(args)
    check_correctness(bench_data)
    plot_performance(bench_data, inference_ctx, args)

    from IPython import embed; embed()
