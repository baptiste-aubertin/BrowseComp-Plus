# Benchmarking Paradigm Search with gpt-oss on BrowseComp-Plus

This guide runs the **LightOn Paradigm** retriever against the BrowseComp-Plus
benchmark, paired with a locally-deployed **gpt-oss-120b** agent served by vLLM.

Retrieval happens remotely over the Paradigm API (`ParadigmSearcher`), so no
local BM25/FAISS index is needed — the only GPU workload is the LLM agent and,
later, the Qwen3-32B judge.

## Components

| Piece | Where it lives | Status |
|-------|----------------|--------|
| `ParadigmSearcher` | `searcher/searchers/paradigm_searcher.py` | search via `POST /api/v3/search`, doc fetch via `/api/v3/files/{id}`; reads `PARADIGM_*` from env |
| Searcher registration | `searcher/searchers/__init__.py` | `SearcherType.PARADIGM = ("paradigm", ...)` |
| Corpus | `data/browsecomp_plus_corpus.jsonl` | 100,195 docs (~3 GB), downloaded via `scripts_paradigm/download_corpus.py` |
| Ingest tooling | `scripts_paradigm/ingest_corpus.py` | idempotent verify / check-failed / per-docid retry into the Paradigm workspace |
| Smoke test | `scripts_paradigm/test_paradigm_searcher.py` | quick `search()` / `get_document()` check against `.env` |

## Prerequisites

`.env` must define the Paradigm credentials (already set for `paradigm-dev`,
workspace `22546`):

```bash
PARADIGM_BASE_URL=https://paradigm-dev.lighton.ai
PARADIGM_API_KEY=...
PARADIGM_WORKSPACE_ID=22546
```

### Corpus ingested into Paradigm

The full corpus must be uploaded **and embedded** in the workspace. Verify (and
fix any gaps) with the idempotent script — rerun until `missing=0 failed=0`:

```bash
python scripts_paradigm/ingest_corpus.py --verify --check-failed
```

Sanity-check retrieval end to end:

```bash
python scripts_paradigm/test_paradigm_searcher.py --query "stuff about cars" --k 5 --fetch-doc
```

### Queries file

`oss_client.py` defaults to `--query topics-qrels/queries.tsv`, which must be
generated from the obfuscated dataset (requires `huggingface-cli login`):

```bash
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

## 1. Deploy gpt-oss-120b (on the H100 node)

gpt-oss-120b ships in **MXFP4** (~63 GB weights) and fits on a single 80 GB
H100. BrowseComp trajectories are long (up to 100 tool-call rounds, growing
context), so the bottleneck is KV-cache — give it 2 GPUs of tensor parallelism
for a comfortable batch and keep the rest of the node free for the judge.

```bash
docker run --gpus '"device=3,7"' -p 8000:8000 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:gptoss \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --max-num-seqs 16
```

- `--gpus '"device=3,7"'` pins the container to host GPUs 3 and 7 (the ordinals
  from `nvidia-smi`). The inner double-quotes are required — `"device=3,7"`
  often fails to parse the comma. Equivalent: `-e NVIDIA_VISIBLE_DEVICES=3,7
  --gpus all`. Inside the container they're renumbered to `0,1`, which is what
  `--tensor-parallel-size 2` uses. If throughput is low, check
  `nvidia-smi topo -m` — a pair on the same NVLink island is faster for TP.
- `-v …huggingface` caches the weights so restarts are instant.
- `--tensor-parallel-size 2` → ~97 GB free for KV across 2 GPUs → ~16 concurrent
  long trajectories. Bump to 4 if vLLM logs show KV evictions/preemptions.
- `--max-num-seqs` should match (or slightly exceed) the bench `--num-threads`.
- To use the **whole** node for max throughput, run
  `--data-parallel-size 4 --tensor-parallel-size 2` (4 replicas × 2 GPUs);
  vLLM load-balances across them on the one port.

### Health check (on the node)

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
```

Wait until it lists `openai/gpt-oss-120b` (first boot downloads the weights).

## 2. Reach the endpoint via SSH tunnel (from your Mac)

```bash
ssh -N -o ClearAllForwardings=yes -L 8009:localhost:8000 Cray3
```

Maps Mac-side port **8009** to the node's `localhost:8000` (8009 avoids clashes
with other local forwards). `ClearAllForwardings=yes` ignores any `LocalForward`
lines in your ssh config's `Host` block so they don't error out. Leave it
running in its own terminal, then sanity-check:

```bash
curl -s http://localhost:8009/v1/models
```

## 3. Run the benchmark (from your Mac)

```bash
python search_agent/oss_client.py \
  --model openai/gpt-oss-120b \
  --model-url http://localhost:8000/v1 \
  --searcher-type paradigm \
  --output-dir runs/paradigm/oss-120b \
  --num-threads 10 \
  --reasoning-effort high
```

- `--model-url http://localhost:8009/v1` points at the SSH tunnel (override
  needed — the default is port 8000).
- Paradigm handles all retrieval over its API — no `--index-path` /
  `--model-name` needed.
- Start at `--num-threads 10`; raise it (and `--max-num-seqs`) if the node is
  underloaded.
- Optional: add `--get-document` to also register the doc-fetch tool
  (`ParadigmSearcher.get_document` is implemented).

## 4. Evaluate (Qwen3-32B judge — on the H100 node)

The judge needs a GPU. Run it on the node (or `scp` the run dir over), giving it
GPUs separate from the ones vLLM is using:

```bash
# pick two free cards — not 3/7, which vLLM is using
CUDA_VISIBLE_DEVICES=2,4 python scripts_evaluation/evaluate_run.py \
  --input_dir runs/paradigm/oss-120b \
  --tensor_parallel_size 2
```

Produces the leaderboard-format JSON: Accuracy / Recall / tool-call stats /
calibration error.

## Retrieval-only evaluation (optional)

To score the Paradigm retriever in isolation (independent of the agent), export
a TREC run file and evaluate against the qrels:

```bash
python -m pyserini.eval.trec_eval -c -m recall.5,100,1000 -m ndcg_cut.10 \
  topics-qrels/qrel_evidence.txt {YOUR_TREC_RUN_FILE}
python -m pyserini.eval.trec_eval -c -m recall.5,100,1000 -m ndcg_cut.10 \
  topics-qrels/qrel_golds.txt {YOUR_TREC_RUN_FILE}
```
