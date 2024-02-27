import os
import json
import subprocess
from argmaxtools import utils, test_utils
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt

logger = utils.get_logger(__name__)

SETUP_CMD = "env CMAKE_BUILD_PARALLEL_LEVEL="" pip install -e ."
MAX_CONTEXT = 16400
MEASURE_EVERY_N_TOKENS = 100
MIN_MISMATCH_AFTER_N_TOKENS = 100
PROMPT = "Continue this series forever: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
BENCH_CMD = "python llms/mistral/mistral.py" + \
    " --model-path ../external/model " + \
    f" --max-tokens {MAX_CONTEXT} " + \
    f" --tokens-per-eval {MEASURE_EVERY_N_TOKENS}" + \
    f" --prompt '{PROMPT}'"

# To compute speed-of-light inference speed
MAC_MEMORY_BW = {
    "M1": 68.3,
    "M1 Pro": 200,
    "M1 Max": 400,
    "M1 Ultra": 800,
    "M2": 100,
    "M2 Pro": 200,
    "M2 Max": 400,
    "M2 Ultra": 800,
    "M3": 100,
    "M3 Pro": 150,
    "M3 Max": {"40": 400, "30": 300},
}


def get_speed_of_light_inference_speed(inference_ctx, model):
    pass


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

    # Install repo B
    logger.info("Installing repo B")
    subprocess.check_call(SETUP_CMD, shell=True, cwd=os.path.join(args.output_dir, "repo_b", args.repo_a.split("/")[1]))

    # Benchmarking --bench-cmd under repo B
    logger.info("Benchmarking repo B")
    subprocess.check_call(
        BENCH_CMD + " --benchmark-json-path benchmark_b.json --optimized-sdpa",
        shell=True, cwd=os.path.join(os.getcwd(), "mlx-examples"))

    # Load benchmark data from repo B
    with open(os.path.join(os.getcwd(), "mlx-examples", "benchmark_b.json"), "r") as f:
        bench_data["repo_b"] = json.load(f)

    # Install repo A
    logger.info("Installing repo A")
    subprocess.check_call(SETUP_CMD, shell=True, cwd=os.path.join(args.output_dir, "repo_a", args.repo_a.split("/")[1]))

    # Benchmarking --bench-cmd under repo A
    logger.info("Benchmarking repo A")
    subprocess.check_call(
        BENCH_CMD + " --benchmark-json-path benchmark_a.json ",
        shell=True, cwd=os.path.join(os.getcwd(), "mlx-examples"))

    # Load benchmark data from repo A
    with open(os.path.join(os.getcwd(), "mlx-examples", "benchmark_a.json"), "r") as f:
        bench_data["repo_a"] = json.load(f)

    return bench_data


def check_correctness(bench_data):
    # Check correctness
    mismatch_after_n_tokens = None
    for idx, (result_a, result_b) in enumerate(zip(bench_data["repo_a"], bench_data["repo_b"])):
        if not result_a[-1] == result_b[-1]:
            logger.error(f"Mismatch in results: {result_a[-1]} vs {result_b[-1]}")
            mismatch_after_n_tokens = idx * MEASURE_EVERY_N_TOKENS
            logger.info(f"First mismatch after n tokens: {mismatch_after_n_tokens}")
            if mismatch_after_n_tokens < MIN_MISMATCH_AFTER_N_TOKENS:
                raise ValueError(
                    f"First mismatch after {mismatch_after_n_tokens} tokens "
                    f"(Less than {MIN_MISMATCH_AFTER_N_TOKENS})")
            break
        else:
            logger.info(f"Results match: {result_a[-1]} vs {result_b[-1]}")
    return mismatch_after_n_tokens


def plot_performance(bench_data, inference_ctx, args):
    # Plot performance
    f, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot([x[0] for x in bench_data["repo_a"]], [x[1] for x in bench_data["repo_a"]], label=f"{args.repo_a}@{args.commit_a[:7]}")
    ax[0].plot([x[0] for x in bench_data["repo_b"]], [x[1] for x in bench_data["repo_b"]], label=f"{args.repo_b}@{args.commit_b[:7]}")
    ax[0].set_xlabel("Context Length (tokens)")
    ax[0].set_ylabel("Inference Speed (tokens/second)")
    ax[0].set_title(
        f"Model={args.hub_model_name} \n"
        f"Device=({inference_ctx['device_spec']['product_name']},"
        f" {inference_ctx['device_spec']['gpu_core_count']} GPU cores, "
        f"macOS={inference_ctx['os_spec']['os_build_number']})"
    )
    ax[0].legend()

    ax[1].set_ylabel("Speedup")
    ax[1].set_xlabel("Context Length (tokens)")
    ax[1].plot(
        [x[0] for x in bench_data["repo_a"]],
        [y[1]/x[1] for x, y in zip(bench_data["repo_a"], bench_data["repo_b"])],
    )

    # TODO(atiorh): Plot speed-of-light inference speed based on memory bandwidth

    f.savefig("bench.png")


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
        choices=(
            "mlx-community/Mistral-7B-Instruct-v0.2",
            "mlx-community/Mistral-7B-Instruct-v0.2-4-bit",
            "mlx-community/Mistral-7B-Instruct-v0.2-8-bit",
        )
    )
    parser.add_argument("--max-context-length", default=MAX_CONTEXT, type=int, help="Maximum context length (in tokens)")

    args = parser.parse_args()

    MAX_CONTEXT = args.max_context_length

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
    mismatch_after_n_tokens = check_correctness(bench_data)
    plot_performance(bench_data, inference_ctx, args)
