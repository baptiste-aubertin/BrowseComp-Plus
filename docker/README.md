# Docker evaluation

Runs the standard LLM-as-judge evaluation (`scripts_evaluation/evaluate_run.py`,
Qwen3-32B judge) inside a container, avoiding host-side issues entirely:

- no Python dev headers needed on the host (triton/torch.compile builds a C
  extension at startup and fails with `fatal error: Python.h` otherwise);
- no NCCL/TCPStore networking problems — inside the container the distributed
  rendezvous happens on the container's own network namespace, so firewalled
  host interfaces (e.g. `bond0`) are never involved;
- no local Python environment required, the vLLM image ships everything.

## Quick start

```bash
# Evaluate a run on GPUs 0,1 (tensor parallel size is inferred from GPUS)
./docker/run_eval.sh --input_dir runs/my_model

# Use more GPUs
GPUS=0,1,2,3 ./docker/run_eval.sh --input_dir runs/my_model

# Any evaluate_run.py flag passes through
./docker/run_eval.sh --input_dir runs/my_model --force --eval_dir ./evals
```

The first call builds the `browsecomp-plus-eval` image automatically.

Results land in `./evals/<run_name>/` on the host (the repo is bind-mounted),
including the leaderboard summary JSON and `detailed_judge_results.csv`.
Already-evaluated queries are skipped on re-runs unless you pass `--force`.

## Caches and mounts

| Host path | Container path | Purpose |
|---|---|---|
| repo root | `/workspace` | code, `runs/`, `evals/`, `data/`, `topics-qrels/` |
| `$HF_CACHE` (default `/mnt/nfs/baptiste_shared/hf_home`) | `/hf_home` | shared HuggingFace model cache (judge weights downloaded once) |

The container runs with your uid/gid so files written to the NFS mounts stay
owned by you.

## vLLM version

`docker/Dockerfile.eval` defaults to `vllm/vllm-openai:v0.9.0`, matching the
`vllm>=0.9.0` pin in `pyproject.toml`. To reuse a vLLM image you already have
locally (skips a ~10 GB pull; the judge API used by the script is stable):

```bash
docker build -f docker/Dockerfile.eval -t browsecomp-plus-eval \
  --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.22.1 docker/
```
