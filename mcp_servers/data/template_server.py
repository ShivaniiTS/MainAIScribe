"""
Template server — loads and selects clinical note templates from config/templates/.

Templates are YAML files defining note structure (sections, prompt hints, formatting).
Selection priority:
    1. Exact match on specialty + visit_type
    2. Specialty match with visit_type=default
    3. soap_default fallback

Usage:
    from mcp_servers.data.template_server import TemplateServer
    server = TemplateServer()
    tpl = server.get_template("orthopedic", "initial_evaluation")
    # tpl.sections -> list of TemplateSection
    # tpl.formatting -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "config" / "templates"


@dataclass
class TemplateSection:
    id: str
    label: str
    required: bool = True
    prompt_hint: str = ""


@dataclass
class NoteTemplate:
    name: str
    specialty: str
    visit_type: str
    header_fields: list[str] = field(default_factory=list)
    sections: list[TemplateSection] = field(default_factory=list)
    formatting: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""


class TemplateServer:
    """Loads all templates from config/templates/ and selects the best match."""

    def __init__(self, templates_dir: Optional[Path] = None) -> None:
        self._dir = templates_dir or _TEMPLATES_DIR
        self._templates: list[NoteTemplate] = []
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return
        import yaml

        for path in sorted(self._dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text())
                tpl = self._parse(data, path.name)
                self._templates.append(tpl)
                logger.debug("template_server: loaded %s", path.name)
            except Exception as exc:
                logger.warning("template_server: failed to load %s — %s", path.name, exc)

        logger.info("template_server: loaded %d templates", len(self._templates))
        self._loaded = True

    @staticmethod
    def _parse(data: dict, source: str) -> NoteTemplate:
        sections = [
            TemplateSection(
                id=s["id"],
                label=s["label"],
                required=s.get("required", True),
                prompt_hint=s.get("prompt_hint", ""),
            )
            for s in data.get("sections", [])
        ]
        return NoteTemplate(
            name=data.get("name", source),
            specialty=data.get("specialty", "general").lower(),
            visit_type=data.get("visit_type", "default").lower(),
            header_fields=data.get("header_fields", []),
            sections=sections,
            formatting=data.get("formatting", {}),
            source_file=source,
        )

    def get_template(
        self,
        specialty: str = "general",
        visit_type: str = "default",
    ) -> NoteTemplate:
        """
        Return the best matching template.

        Priority:
            1. Exact specialty + visit_type match
            2. Specialty match (any visit_type)
            3. soap_default
        """
        self._load_all()
        spec = specialty.lower().strip()
        vtype = visit_type.lower().strip().replace(" ", "_")

        # 1. Exact match
        for tpl in self._templates:
            if tpl.specialty == spec and tpl.visit_type == vtype:
                return tpl

        # 2. Specialty match
        for tpl in self._templates:
            if tpl.specialty == spec:
                return tpl

        # 3. soap_default
        for tpl in self._templates:
            if tpl.source_file == "soap_default.yaml":
                return tpl

        # Last resort: synthetic default
        logger.warning("template_server: no template found for %s/%s — using empty default", spec, vtype)
        return NoteTemplate(
            name="Fallback SOAP",
            specialty="general",
            visit_type="default",
            sections=[
                TemplateSection("subjective", "SUBJECTIVE"),
                TemplateSection("objective", "OBJECTIVE"),
                TemplateSection("assessment", "ASSESSMENT"),
                TemplateSection("plan", "PLAN"),
            ],
        )

    def list_templates(self) -> list[NoteTemplate]:
        self._load_all()
        return list(self._templates)


# Module-level singleton
_server: Optional[TemplateServer] = None


def get_template_server() -> TemplateServer:
    global _server
    if _server is None:
        _server = TemplateServer()
    return _server
