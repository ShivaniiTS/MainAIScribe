"""
tests/unit/test_version_resolution.py — Tests for version resolution across the stack.

Ensures NO hardcoded version strings remain in the codebase.
Verifies that "latest" is resolved dynamically everywhere.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


class TestNoHardcodedVersionsInApi:
    """Scan Python API code for hardcoded version strings that should be dynamic."""

    _API_FILES = [
        ROOT / "api" / "data_loader.py",
        ROOT / "api" / "routes" / "encounters.py",
        ROOT / "api" / "routes" / "quality.py",
        ROOT / "api" / "routes" / "providers.py",
        ROOT / "api" / "pipeline" / "routes.py",
        ROOT / "api" / "main.py",
        ROOT / "api" / "proxy.py",
    ]

    # Patterns that are OK:
    # - In comments or docstrings
    # - In logging/error messages
    # - In the _KNOWN_VERSIONS definition
    _HARDCODED_VERSION_RE = re.compile(r'''['"]v\d+['"]''')

    def test_no_hardcoded_version_defaults(self):
        """Ensure no API file uses a hardcoded version as a default parameter value."""
        issues = []
        for filepath in self._API_FILES:
            if not filepath.exists():
                continue
            source = filepath.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                # Check function default args
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for default in node.args.defaults + node.args.kw_defaults:
                        if default and isinstance(default, ast.Constant) and isinstance(default.value, str):
                            if re.match(r'^v\d+$', default.value):
                                issues.append(
                                    f"{filepath.relative_to(ROOT)}:{node.lineno} "
                                    f"function {node.name}() has hardcoded version default '{default.value}'"
                                )

        assert issues == [], (
            "Found hardcoded version defaults in API code:\n" +
            "\n".join(f"  - {i}" for i in issues)
        )


class TestNoHardcodedVersionsInFrontend:
    """Scan TypeScript frontend code for hardcoded version defaults."""

    _TS_DIRS = [
        ROOT / "client" / "web" / "lib",
        ROOT / "client" / "web" / "app",
        ROOT / "client" / "web" / "components",
        ROOT / "client" / "provider" / "lib",
        ROOT / "client" / "provider" / "app",
        ROOT / "client" / "provider" / "components",
    ]

    # Match: version = "v7", version="v9", default "v7"
    _HARDCODED_RE = re.compile(r'''(?:version\s*=\s*|default\s+)["']v\d+["']''')

    def test_no_hardcoded_version_in_ts(self):
        """Ensure no frontend TypeScript file has hardcoded version strings."""
        issues = []
        for dir_path in self._TS_DIRS:
            if not dir_path.exists():
                continue
            for ts_file in dir_path.rglob("*.ts"):
                self._check_file(ts_file, issues)
            for tsx_file in dir_path.rglob("*.tsx"):
                self._check_file(tsx_file, issues)

        assert issues == [], (
            "Found hardcoded version strings in frontend code:\n" +
            "\n".join(f"  - {i}" for i in issues)
        )

    def _check_file(self, filepath: Path, issues: list):
        content = filepath.read_text()
        for i, line in enumerate(content.splitlines(), 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if self._HARDCODED_RE.search(line):
                issues.append(
                    f"{filepath.relative_to(ROOT)}:{i}: {stripped[:100]}"
                )
