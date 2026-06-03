"""LightOn Paradigm Console searcher: POST /api/v3/search + GET /api/v3/files/{id}."""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from .base import BaseSearcher

logger = logging.getLogger(__name__)

PARADIGM_MAX_RESULTS = 50


class ParadigmSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--base-url",
            default=os.getenv("PARADIGM_BASE_URL"),
            help="Paradigm Console base URL (default: $PARADIGM_BASE_URL).",
        )
        parser.add_argument(
            "--api-key",
            default=os.getenv("PARADIGM_API_KEY"),
            help="Paradigm API key (default: $PARADIGM_API_KEY).",
        )
        parser.add_argument(
            "--workspace-id",
            type=int,
            default=int(os.environ["PARADIGM_WORKSPACE_ID"])
            if os.getenv("PARADIGM_WORKSPACE_ID")
            else None,
            help="Restrict search to this workspace (default: $PARADIGM_WORKSPACE_ID, else cross-workspace).",
        )
        parser.add_argument(
            "--mode",
            choices=["text", "vision"],
            default="text",
            help="Search mode: 'text' (hybrid) or 'vision' (VLM page-image). Default: text.",
        )
        parser.add_argument(
            "--skip-rerank",
            action="store_true",
            default=False,
            help="Skip reranking for lower latency (score.reranking will be null).",
        )
        parser.add_argument(
            "--request-timeout",
            type=float,
            default=120.0,
            help="HTTP timeout in seconds for /search and /files calls (default: 120).",
        )

    def __init__(self, args):
        self.args = args

        if not args.base_url:
            raise ValueError("paradigm --base-url is required (or set $PARADIGM_BASE_URL)")
        if not args.api_key:
            raise ValueError("paradigm --api-key is required (or set $PARADIGM_API_KEY)")

        self.base_url = args.base_url.rstrip("/")
        self.workspace_id = args.workspace_id
        self.mode = args.mode
        self.skip_rerank = args.skip_rerank
        self.timeout = args.request_timeout

        self.session = requests.Session()
        self.session.headers["X-Api-Key"] = args.api_key

        # docid (external_id) -> file_id, populated by search results and on-demand lookups.
        self._docid_to_file_id: Dict[str, int] = {}

        logger.info(
            "Paradigm searcher ready (base_url=%s workspace_id=%s mode=%s skip_rerank=%s)",
            self.base_url,
            self.workspace_id,
            self.mode,
            self.skip_rerank,
        )

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        # Always pull the API's maximum candidate pool so the reranker sees as many
        # chunks as possible; then dedup by docid (best chunk per doc) and take top-k
        # unique docids. Billing is one search credit regardless of max_results.
        payload: Dict[str, Any] = {
            "query": query,
            "max_results": PARADIGM_MAX_RESULTS,
            "mode": self.mode,
            "skip_rerank": self.skip_rerank,
        }
        if self.workspace_id is not None:
            payload["workspace_id"] = [self.workspace_id]

        r = self.session.post(f"{self.base_url}/api/v3/search", json=payload, timeout=self.timeout)
        r.raise_for_status()
        results = r.json().get("results", [])

        # Keep the highest-scoring chunk per docid (explicit max, independent of response order),
        # then sort docids by best-chunk score and take top-k.
        best_per_docid: Dict[str, Dict[str, Any]] = {}
        for hit in results:
            source = hit.get("source") or {}
            ext_meta = source.get("external_metadata") or {}
            docid = ext_meta.get("external_id")
            if not docid:
                # Fall back to file_id when external_id is missing (e.g. directly-uploaded docs).
                file_id = source.get("file_id")
                if file_id is None:
                    continue
                docid = str(file_id)
            docid = str(docid)

            file_id = source.get("file_id")
            if file_id is not None:
                self._docid_to_file_id[docid] = int(file_id)

            score = hit.get("score") or {}
            score_value = score.get("reranking")
            if score_value is None:
                score_value = score.get("retrieval", 0.0)
            score_value = float(score_value) if score_value is not None else 0.0

            existing = best_per_docid.get(docid)
            if existing is not None and existing["score"] >= score_value:
                continue

            best_per_docid[docid] = {
                "docid": docid,
                "score": score_value,
                "text": hit.get("content") or "",
            }

        ranked = sorted(best_per_docid.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]

    def _resolve_file_id(self, docid: str) -> Optional[int]:
        cached = self._docid_to_file_id.get(docid)
        if cached is not None:
            return cached

        params: Dict[str, Any] = {
            "external_metadata__external_id": docid,
            "page_size": 50,
        }
        if self.workspace_id is not None:
            params["workspace_id"] = self.workspace_id

        r = self.session.get(f"{self.base_url}/api/v3/files", params=params, timeout=self.timeout)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None

        # Prefer an embedded file; otherwise take the first match.
        chosen = next((f for f in results if f.get("status") == "embedded"), results[0])
        file_id = int(chosen["id"])
        self._docid_to_file_id[docid] = file_id
        return file_id

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        file_id = self._resolve_file_id(docid)
        if file_id is None:
            return None

        r = self.session.get(
            f"{self.base_url}/api/v3/files/{file_id}",
            params={"include_content": "true"},
            timeout=self.timeout,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        body = r.json()
        text = body.get("content")
        if text is None:
            return None
        return {"docid": docid, "text": text}

    @property
    def search_type(self) -> str:
        return "Paradigm-Console"

    def search_description(self, k: int = 10) -> str:
        return (
            f"Perform a hybrid retrieval search against the LightOn Paradigm knowledge base. "
            f"Returns up to {k} unique documents with docid, score, and snippet "
            f"(the highest-scoring chunk's text content for each document)."
        )

    def get_document_description(self) -> str:
        return "Retrieve the full text content of a document by its docid from Paradigm."
