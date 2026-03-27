"""
Medical dictionary server — loads the base 98K medical wordlist and
specialty-specific term lists.

Usage:
    from mcp_servers.data.medical_dict_server import MedicalDictServer
    srv = MedicalDictServer()
    terms = srv.get_terms("orthopedic")   # base + orthopedic combined
    hotwords = srv.get_hotwords("orthopedic", max_terms=200)  # ASR hotwords
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_BASE_WORDLIST = Path(__file__).parent.parent.parent / "postprocessor" / "medical_wordlist.txt"
_DICT_DIR = Path(__file__).parent.parent.parent / "config" / "dictionaries"

# Specialty name → dictionary filename
_SPECIALTY_FILES: dict[str, str] = {
    "orthopedic":        "orthopedic.txt",
    "ortho":             "orthopedic.txt",
    "gastroenterology":  "gastroenterology.txt",
    "gi":                "gastroenterology.txt",
    "cardiology":        "cardiology.txt",
    "neurology":         "neurology.txt",
    "psychiatry":        "psychiatry.txt",
    "pediatrics":        "pediatrics.txt",
    "general":           None,
}


class MedicalDictServer:
    """Loads base + specialty medical term dictionaries."""

    def __init__(self, dict_dir: Optional[Path] = None) -> None:
        self._dir = dict_dir or _DICT_DIR
        self._cache: dict[str, set[str]] = {}

    def _load_file(self, path: Path) -> set[str]:
        if not path.exists():
            logger.debug("medical_dict_server: %s not found, skipping", path)
            return set()
        terms: set[str] = set()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            word = line.strip().lower()
            if word and not word.startswith("#"):
                terms.add(word)
        return terms

    def get_terms(self, specialty: str = "general") -> set[str]:
        """
        Return combined base + specialty term set.

        Results are cached per specialty.
        """
        key = specialty.lower()
        if key in self._cache:
            return self._cache[key]

        base = self._load_file(_BASE_WORDLIST)
        logger.info(
            "medical_dict_server: base wordlist %d terms", len(base)
        )

        spec_file_name = _SPECIALTY_FILES.get(key)
        if spec_file_name:
            spec_path = self._dir / spec_file_name
            spec_terms = self._load_file(spec_path)
            logger.info(
                "medical_dict_server: %s specialty %d terms", key, len(spec_terms)
            )
            combined = base | spec_terms
        else:
            combined = base

        self._cache[key] = combined
        return combined

    def get_hotwords(self, specialty: str = "general", max_terms: int = 200) -> list[str]:
        """
        Return specialty-specific hotwords for ASR boosting.

        Returns only the specialty terms (not base), sorted by length desc
        (longer = more specific = higher priority for ASR boosting).
        """
        spec_file_name = _SPECIALTY_FILES.get(specialty.lower())
        if not spec_file_name:
            return []
        spec_path = self._dir / spec_file_name
        terms = self._load_file(spec_path)
        # Longer terms are more specific — prioritise for hotword injection
        return sorted(terms, key=len, reverse=True)[:max_terms]

    def get_specialty_context(self, specialty: str, max_terms: int = 50) -> str:
        """
        Return a compact specialty vocabulary string for LLM prompt injection.

        Example: "Orthopedic vocabulary: cervical, ROM, Spurling, Tinel, ..."
        """
        hotwords = self.get_hotwords(specialty, max_terms=max_terms)
        if not hotwords:
            return ""
        label = specialty.title()
        return f"{label} vocabulary: {', '.join(hotwords[:max_terms])}"


# Module-level singleton
_server: Optional[MedicalDictServer] = None


def get_dict_server() -> MedicalDictServer:
    global _server
    if _server is None:
        _server = MedicalDictServer()
    return _server
