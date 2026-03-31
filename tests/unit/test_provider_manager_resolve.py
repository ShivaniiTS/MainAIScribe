from __future__ import annotations

from config.provider_manager import ProviderManager


def _write_provider_yaml(path, provider_id: str, name: str) -> None:
    path.write_text(
        "\n".join([
            f"id: {provider_id}",
            f"name: {name}",
            "specialty: chiropractic",
            "template_routing:",
            "  default: chiro_follow_up",
        ]) + "\n",
        encoding="utf-8",
    )


def test_resolve_provider_id_exact_id(tmp_path):
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir()
    _write_provider_yaml(providers_dir / "dr_mohammed_alwahaidy.yaml", "dr_mohammed_alwahaidy", "Dr. Mohammed Alwahaidy")

    mgr = ProviderManager(providers_dir=providers_dir)
    assert mgr.resolve_provider_id("dr_mohammed_alwahaidy") == "dr_mohammed_alwahaidy"


def test_resolve_provider_id_from_last_first_name(tmp_path):
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir()
    _write_provider_yaml(providers_dir / "dr_mohammed_alwahaidy.yaml", "dr_mohammed_alwahaidy", "Dr. Mohammed Alwahaidy")

    mgr = ProviderManager(providers_dir=providers_dir)
    assert mgr.resolve_provider_id("Alwahaidy, Mohammed") == "dr_mohammed_alwahaidy"


def test_resolve_provider_id_from_numeric_id_plus_name(tmp_path):
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir()
    _write_provider_yaml(providers_dir / "dr_mohammed_alwahaidy.yaml", "dr_mohammed_alwahaidy", "Dr. Mohammed Alwahaidy")

    mgr = ProviderManager(providers_dir=providers_dir)
    assert mgr.resolve_provider_id("300", provider_name="Alwahaidy, Mohammed") == "dr_mohammed_alwahaidy"


def test_resolve_provider_id_no_match_falls_back_to_input(tmp_path):
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir()
    _write_provider_yaml(providers_dir / "dr_mohammed_alwahaidy.yaml", "dr_mohammed_alwahaidy", "Dr. Mohammed Alwahaidy")

    mgr = ProviderManager(providers_dir=providers_dir)
    assert mgr.resolve_provider_id("unknown_provider") == "unknown_provider"
