"""
Searchers package for different search implementations.

Searcher classes are imported lazily (only when their CLI name is selected) so
that picking one searcher does not drag in the heavy / optional dependencies of
the others (e.g. pyserini+Java for BM25, pylate, faiss).
"""

import importlib
from enum import Enum

from .base import BaseSearcher


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings.

    Each value is ``(cli_name, module_name, class_name)``; the class is resolved
    on demand via :attr:`searcher_class`.
    """

    BM25 = ("bm25", "bm25_searcher", "BM25Searcher")
    FAISS = ("faiss", "faiss_searcher", "FaissSearcher")
    REASONIR = ("reasonir", "faiss_searcher", "ReasonIrSearcher")
    PYLATE = ("pylate", "pylate_searcher", "PylateSearcher")
    PARADIGM = ("paradigm", "paradigm_searcher", "ParadigmSearcher")
    CUSTOM = ("custom", "custom_searcher", "CustomSearcher")  # yet to be implemented

    def __init__(self, cli_name, module_name, class_name):
        self.cli_name = cli_name
        self._module_name = module_name
        self._class_name = class_name

    @property
    def searcher_class(self):
        """Import the searcher's module on demand and return its class."""
        module = importlib.import_module(f".{self._module_name}", __package__)
        return getattr(module, self._class_name)

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name (imported lazily)."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                return searcher_type.searcher_class
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = ["BaseSearcher", "SearcherType"]
