#!/usr/bin/env bash
# Run the LLM-as-judge evaluation inside Docker.
#
# Usage:
#   ./docker/run_eval.sh --input_dir runs/my_model [evaluate_run.py args...]
#
# Environment overrides:
#   GPUS         GPU ids to expose (default: 0,1). Tensor parallel size
#                defaults to the number of GPUs listed; override by passing
#                --tensor_parallel_size explicitly.
#   HF_CACHE     Host HuggingFace cache dir (default: /mnt/nfs/baptiste_shared/hf_home)
#   IMAGE        Docker image to use (default: browsecomp-plus-eval, built on demand)
#
# Examples:
#   ./docker/run_eval.sh --input_dir runs/paradigm/oss-120b
#   GPUS=0,1,2,3 ./docker/run_eval.sh --input_dir runs/paradigm/oss-120b
set -euo pipefail

GPUS="${GPUS:-0,1}"
HF_CACHE="${HF_CACHE:-/mnt/nfs/baptiste_shared/hf_home}"
IMAGE="${IMAGE:-browsecomp-plus-eval}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found, building it..."
    docker build -f "$REPO_DIR/docker/Dockerfile.eval" -t "$IMAGE" "$REPO_DIR/docker"
fi

# Default tensor parallel size = number of GPUs exposed, unless caller sets it.
TP_ARGS=()
if [[ "$*" != *"--tensor_parallel_size"* ]]; then
    NUM_GPUS=$(awk -F, '{print NF}' <<<"$GPUS")
    TP_ARGS=(--tensor_parallel_size "$NUM_GPUS")
fi

mkdir -p "$HF_CACHE"

# --ipc=host: NCCL/vLLM workers need large shared memory segments.
# --user + HOME=/tmp: files written to the NFS mounts keep your uid, while
#   triton/torch compile caches go to a writable location inside the container.
# USER/LOGNAME: our uid has no /etc/passwd entry in the container, and torch
#   resolves the cache dir via getpass.getuser(), which reads these first.
exec docker run --rm -it \
    --gpus "\"device=${GPUS}\"" \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e USER="$(id -un)" \
    -e LOGNAME="$(id -un)" \
    -e HF_HOME=/hf_home \
    -v "$REPO_DIR":/workspace \
    -v "$HF_CACHE":/hf_home \
    -w /workspace \
    "$IMAGE" \
    python3 scripts_evaluation/evaluate_run.py "${TP_ARGS[@]}" "$@"
