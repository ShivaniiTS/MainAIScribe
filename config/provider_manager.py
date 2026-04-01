"""
Provider profile manager — YAML-based CRUD for ProviderProfile objects.

Profiles are stored as individual YAML files under config/providers/.
One file per provider, named {provider_id}.yaml.

Usage:
    from config.provider_manager import get_provider_manager
    mgr = get_provider_manager()
    profile = mgr.load("dr_faraz_rahman")
    mgr.save(profile)                      # persist changes
    mgr.update_quality_score(             # after a quality sweep
        provider_id="dr_faraz_rahman",
        version="v3",
        score=4.34,
        sample_count=22,
    )
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from orchestrator.state import NoteType, ProviderProfile

logger = logging.getLogger(__name__)

_PROVIDERS_DIR = Path(__file__).parent / "providers"


class ProviderManager:
    """YAML-backed CRUD for ProviderProfile objects."""

    def __init__(self, providers_dir: Optional[Path] = None) -> None:
        self._dir = providers_dir or _PROVIDERS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ProviderProfile] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Load / Save
    # ──────────────────────────────────────────────────────────────────────

    def load(self, provider_id: str) -> ProviderProfile:
        """
        Load a provider profile from YAML.

        Raises FileNotFoundError if the profile does not exist.
        """
        if provider_id in self._cache:
            return self._cache[provider_id]

        path = self._dir / f"{provider_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No provider profile for '{provider_id}' at {path}")

        data = yaml.safe_load(path.read_text())
        profile = self._from_dict(data)
        self._cache[provider_id] = profile
        logger.debug("provider_manager: loaded profile %s", provider_id)
        return profile

    def load_or_default(self, provider_id: str) -> ProviderProfile:
        """Load profile or return a minimal default if not found."""
        try:
            return self.load(provider_id)
        except FileNotFoundError:
            logger.warning("provider_manager: no profile for %s — using defaults", provider_id)
            return ProviderProfile(id=provider_id, name=provider_id, specialty="general")

    def save(self, profile: ProviderProfile) -> None:
        """Persist a provider profile to YAML, merging extra fields."""
        path = self._dir / f"{profile.id}.yaml"

        # Preserve extra fields that are in the YAML but not in ProviderProfile
        existing: dict = {}
        if path.exists():
            existing = yaml.safe_load(path.read_text()) or {}

        # Build updated dict: existing fields + Pydantic fields
        updated = {**existing, **self._to_dict(profile)}
        path.write_text(yaml.dump(updated, default_flow_style=False, allow_unicode=True))
        self._cache[profile.id] = profile
        logger.debug("provider_manager: saved profile %s", profile.id)

    def list_providers(self) -> list[str]:
        """Return all provider IDs that have a profile YAML."""
        return [p.stem for p in sorted(self._dir.glob("*.yaml"))]

    def create(self, profile: ProviderProfile, overwrite: bool = False) -> None:
        """Create a new provider profile. Raises if already exists unless overwrite=True."""
        path = self._dir / f"{profile.id}.yaml"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Profile '{profile.id}' already exists. Use overwrite=True.")
        self.save(profile)

    def delete(self, provider_id: str) -> None:
        """Delete a provider profile YAML."""
        path = self._dir / f"{provider_id}.yaml"
        if path.exists():
            path.unlink()
            self._cache.pop(provider_id, None)
        else:
            raise FileNotFoundError(f"No profile for '{provider_id}'")

    # ──────────────────────────────────────────────────────────────────────
    # Quality tracking
    # ──────────────────────────────────────────────────────────────────────

    def update_quality_score(
        self,
        provider_id: str,
        version: str,
        score: float,
        sample_count: int,
        dimension_scores: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Record a quality score for a provider.

        Updates provider_profile.quality_scores and appends to quality_history
        in the YAML. Does NOT require the profile to be in-memory first.
        """
        path = self._dir / f"{provider_id}.yaml"
        if not path.exists():
            logger.warning("provider_manager: no profile for %s, cannot update quality", provider_id)
            return

        data = yaml.safe_load(path.read_text()) or {}

        # quality_scores: {version → score}
        if "quality_scores" not in data or not isinstance(data["quality_scores"], dict):
            data["quality_scores"] = {}
        data["quality_scores"][version] = round(score, 3)

        # quality_history: list of {date, version, score, samples, dimensions}
        if "quality_history" not in data or not isinstance(data["quality_history"], list):
            data["quality_history"] = []
        entry: dict = {
            "date": date.today().isoformat(),
            "version": version,
            "score": round(score, 3),
            "samples": sample_count,
        }
        if dimension_scores:
            entry["dimensions"] = {k: round(v, 3) for k, v in dimension_scores.items()}
        data["quality_history"].append(entry)

        path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
        # Invalidate cache so next load picks up updated scores
        self._cache.pop(provider_id, None)
        logger.info(
            "provider_manager: quality score updated %s %s → %.3f (%d samples)",
            provider_id, version, score, sample_count,
        )

    def get_quality_trend(self, provider_id: str) -> list[dict]:
        """Return quality history for a provider (oldest first)."""
        path = self._dir / f"{provider_id}.yaml"
        if not path.exists():
            return []
        data = yaml.safe_load(path.read_text()) or {}
        return data.get("quality_history", [])

    # ──────────────────────────────────────────────────────────────────────
    # Template routing
    # ──────────────────────────────────────────────────────────────────────

    def resolve_template(self, provider_id: str, visit_type: Optional[str]) -> str:
        """
        Determine the template_id for a given provider + visit type.

        Resolution order:
        1. YAML template_routing table (visit_type → template_id)
        2. Provider's default template_id
        3. Hardcoded fallback: "soap_default"
        """
        path = self._dir / f"{provider_id}.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            routing: dict = data.get("template_routing", {})
            if visit_type:
                # Normalise: lowercase, underscores
                key = visit_type.lower().replace(" ", "_").replace("-", "_")
                if key in routing:
                    return routing[key]
            if "default" in routing:
                return routing["default"]

        profile = self.load_or_default(provider_id)
        return profile.template_id or "soap_default"

    @staticmethod
    def _name_tokens(value: str) -> set[str]:
        """Tokenize provider labels for robust matching.

        Handles values like:
        - "dr_mohammed_alwahaidy"
        - "Dr. Mohammed Alwahaidy"
        - "Alwahaidy, Mohammed"
        """
        if not value:
            return set()
        cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        tokens = {t for t in cleaned.split() if t}
        # Ignore common title/credential tokens in fuzzy matching.
        stop = {"dr", "doctor", "md", "do", "dc", "pa", "np"}
        return {t for t in tokens if t not in stop}

    def resolve_provider_id(self, provider_id: str | None, provider_name: str | None = None) -> str:
        """Resolve an incoming provider identifier to a configured provider profile ID.

        Prefers exact profile ID matches; falls back to fuzzy matching against profile
        names (e.g. "Alwahaidy, Mohammed" -> "dr_mohammed_alwahaidy").
        """
        raw_id = (provider_id or "").strip()
        raw_name = (provider_name or "").strip()

        if not raw_id and not raw_name:
            return ""

        known_ids = self.list_providers()

        # Direct ID match wins immediately.
        if raw_id in known_ids:
            return raw_id
        if raw_name in known_ids:
            return raw_name

        # Build candidate profile token sets once.
        candidates: list[tuple[str, set[str], set[str]]] = []
        for pid in known_ids:
            profile = self.load_or_default(pid)
            candidates.append((pid, self._name_tokens(pid), self._name_tokens(profile.name or "")))

        search_values = [v for v in [raw_id, raw_name] if v]

        # Exact normalized token match against profile id/name.
        for value in search_values:
            vt = self._name_tokens(value)
            if not vt:
                continue
            for pid, id_tokens, name_tokens in candidates:
                if vt == id_tokens or vt == name_tokens:
                    return pid

        # Fuzzy overlap fallback.
        best_id: str | None = None
        best_score = 0.0
        for value in search_values:
            vt = self._name_tokens(value)
            if not vt:
                continue
            for pid, id_tokens, name_tokens in candidates:
                for target_tokens in (id_tokens, name_tokens):
                    if not target_tokens:
                        continue
                    overlap = len(vt & target_tokens)
                    score = overlap / max(len(vt), len(target_tokens))
                    if overlap >= 2 and score > best_score:
                        best_score = score
                        best_id = pid

        if best_id and best_score >= 0.6:
            return best_id

        # No profile match: preserve caller-provided identifier.
        return raw_id or raw_name

    # ──────────────────────────────────────────────────────────────────────
    # Serialization helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _from_dict(data: dict) -> ProviderProfile:
        """Build a ProviderProfile from a raw YAML dict."""
        # note_format: accept both enum values and string keys
        nf_raw = data.get("note_format", "SOAP")
        note_format = NoteType(nf_raw) if nf_raw in {e.value for e in NoteType} else NoteType.SOAP

        # quality_scores: keep dict as-is; template_id from template_routing.default
        routing: dict = data.get("template_routing", {})
        template_id = routing.get("default", data.get("template_id", "soap_default"))

        return ProviderProfile(
            id=data.get("id", "unknown"),
            name=data.get("name", ""),
            credentials=data.get("credentials"),
            specialty=data.get("specialty") or "general",
            npi=data.get("npi"),
            practice_id=data.get("practice_id"),
            note_format=note_format,
            template_id=template_id,
            style_directives=data.get("style_directives", []),
            custom_vocabulary=data.get("custom_vocabulary", []),
            asr_override=data.get("asr_override"),
            llm_override=data.get("llm_override"),
            noise_suppression_level=data.get("noise_suppression_level", "moderate"),
            postprocessor_mode=data.get("postprocessor_mode", "hybrid"),
            style_model_version=data.get("style_model_version", "v0"),
            correction_count=data.get("correction_count", 0),
            quality_scores=data.get("quality_scores") or {},
        )

    @staticmethod
    def _to_dict(profile: ProviderProfile) -> dict:
        """Serialise a ProviderProfile to a YAML-friendly dict."""
        return {
            "id": profile.id,
            "name": profile.name,
            "credentials": profile.credentials,
            "specialty": profile.specialty,
            "npi": profile.npi,
            "practice_id": profile.practice_id,
            "note_format": profile.note_format.value,
            "template_id": profile.template_id,
            "style_directives": profile.style_directives,
            "custom_vocabulary": profile.custom_vocabulary,
            "asr_override": profile.asr_override,
            "llm_override": profile.llm_override,
            "noise_suppression_level": profile.noise_suppression_level,
            "postprocessor_mode": profile.postprocessor_mode,
            "style_model_version": profile.style_model_version,
            "correction_count": profile.correction_count,
            "quality_scores": profile.quality_scores,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    global _manager
    if _manager is None:
        _manager = ProviderManager()
    return _manager
