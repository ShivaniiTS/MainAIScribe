"""
tests/unit/test_cascade_validation.py — Template delete cascade, provider conflict,
and specialty audit consistency tests.

Covers:
1. Template delete blocked (409) when providers reference it
2. Template delete succeeds when no providers reference it
3. Provider creation validates: duplicate ID (409), missing specialty dict (422),
   invalid template routing (422)
4. Provider update validates: missing specialty dict (422), invalid template routing (422)
5. Specialty audit detects: missing dictionaries, specialty mismatches, broken template routing
6. Template creation validates: missing specialty dictionary (422), duplicate ID (409)
"""
from __future__ import annotations

import os
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient


@pytest.fixture
def test_env(tmp_path):
    """Set up isolated config directories with test data."""
    # Create directory structure
    dict_dir = tmp_path / "config" / "dictionaries"
    dict_dir.mkdir(parents=True)
    template_dir = tmp_path / "config" / "templates"
    template_dir.mkdir(parents=True)
    provider_dir = tmp_path / "config" / "providers"
    provider_dir.mkdir(parents=True)

    # Create a specialty dictionary
    (dict_dir / "orthopedic.txt").write_text("# Orthopedic\nROM\nacetabulum\n")
    (dict_dir / "neurology.txt").write_text("# Neurology\nEEG\ncranial\n")

    # Create a template
    template_data = {
        "name": "Ortho Initial",
        "specialty": "orthopedic",
        "visit_type": "initial_evaluation",
        "header_fields": ["patient_name"],
        "sections": [{"id": "chief_complaint", "label": "Chief Complaint", "required": True, "prompt_hint": ""}],
        "formatting": {"voice": "active", "tense": "past", "person": "third",
                       "abbreviations": "spell_out", "measurements": "include_units"},
    }
    with open(template_dir / "ortho_initial_eval.yaml", "w") as f:
        yaml.dump(template_data, f)

    # Create a provider that references the template
    provider_data = {
        "id": "dr_smith",
        "name": "Dr. Smith",
        "credentials": "MD",
        "specialty": "orthopedic",
        "template_routing": {"initial_evaluation": "ortho_initial_eval"},
        "style_directives": [],
        "custom_vocabulary": [],
        "quality_scores": {},
        "quality_history": [],
    }
    with open(provider_dir / "dr_smith.yaml", "w") as f:
        yaml.dump(provider_data, f)

    return tmp_path


@pytest.fixture
def app_client(test_env):
    """Create a TestClient with patched config directories."""
    # Patch the module-level directory paths
    with patch("api.routes.templates.TEMPLATE_DIR", test_env / "config" / "templates"), \
         patch("api.routes.templates.PROVIDER_DIR", test_env / "config" / "providers"), \
         patch("api.routes.specialties.DICT_DIR", test_env / "config" / "dictionaries"), \
         patch("api.routes.providers.PROVIDER_DIR", test_env / "config" / "providers"):

        from fastapi import FastAPI
        from api.routes import templates, specialties, providers
        from api import data_loader as dl

        app = FastAPI()
        app.include_router(templates.router)
        app.include_router(specialties.router)
        app.include_router(providers.router)

        # Mock data_loader functions that providers.router uses
        with patch.object(dl, "list_providers", return_value=[]), \
             patch.object(dl, "get_provider", return_value=None):
            yield TestClient(app)


# ---------------------------------------------------------------------------
# Template delete cascade tests
# ---------------------------------------------------------------------------

class TestTemplateDeleteCascade:
    def test_delete_blocked_when_provider_references(self, app_client, test_env):
        """409 Conflict when a provider routes to the template."""
        resp = app_client.delete("/templates/ortho_initial_eval")
        assert resp.status_code == 409
        assert "used by providers" in resp.json()["detail"]
        assert "dr_smith" in resp.json()["detail"]

    def test_delete_succeeds_when_no_providers_reference(self, app_client, test_env):
        """204 No Content when no providers reference the template."""
        # Create an orphan template
        tpl = {
            "name": "Orphan Template",
            "specialty": "orthopedic",
            "visit_type": "follow_up",
            "header_fields": [],
            "sections": [],
            "formatting": {},
        }
        tpl_path = test_env / "config" / "templates" / "orphan_tpl.yaml"
        with open(tpl_path, "w") as f:
            yaml.dump(tpl, f)

        resp = app_client.delete("/templates/orphan_tpl")
        assert resp.status_code == 204
        assert not tpl_path.exists()

    def test_delete_nonexistent_template_404(self, app_client):
        resp = app_client.delete("/templates/nonexistent_tpl")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Template creation validation tests
# ---------------------------------------------------------------------------

class TestTemplateCreateValidation:
    def test_create_duplicate_id_409(self, app_client):
        resp = app_client.post("/templates", json={
            "id": "ortho_initial_eval",
            "name": "Duplicate",
            "specialty": "orthopedic",
            "visit_type": "initial_evaluation",
        })
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_create_missing_specialty_dict_422(self, app_client):
        resp = app_client.post("/templates", json={
            "id": "cardio_new",
            "name": "Cardio New",
            "specialty": "cardiology",  # No dictionary file exists
            "visit_type": "initial_evaluation",
        })
        assert resp.status_code == 422
        assert "no dictionary" in resp.json()["detail"].lower()

    def test_create_valid_template_201(self, app_client, test_env):
        resp = app_client.post("/templates", json={
            "id": "ortho_follow_up",
            "name": "Ortho Follow-Up",
            "specialty": "orthopedic",
            "visit_type": "follow_up",
            "header_fields": ["patient_name", "date_of_service"],
            "sections": [
                {"id": "chief_complaint", "label": "Chief Complaint", "required": True, "prompt_hint": ""},
                {"id": "assessment", "label": "Assessment", "required": True, "prompt_hint": ""},
            ],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "ortho_follow_up"
        assert data["section_count"] == 2

    def test_create_empty_specialty_skips_validation(self, app_client, test_env):
        """Empty specialty string should not trigger dictionary validation."""
        resp = app_client.post("/templates", json={
            "id": "generic_template",
            "name": "Generic Template",
            "specialty": "",
            "visit_type": "follow_up",
        })
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Provider creation validation tests
# ---------------------------------------------------------------------------

class TestProviderCreateValidation:
    def test_create_duplicate_id_409(self, app_client, test_env):
        resp = app_client.post("/providers", json={
            "id": "dr_smith",
            "name": "Dr. Smith Clone",
            "specialty": "orthopedic",
        })
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_create_missing_specialty_dict_422(self, app_client):
        resp = app_client.post("/providers", json={
            "id": "dr_new",
            "name": "Dr. New",
            "specialty": "dermatology",  # No dictionary file
        })
        assert resp.status_code == 422
        assert "no dictionary" in resp.json()["detail"].lower()

    def test_create_invalid_template_routing_422(self, app_client):
        resp = app_client.post("/providers", json={
            "id": "dr_bad_route",
            "name": "Dr. Bad Route",
            "specialty": "orthopedic",
            "template_routing": {"initial_evaluation": "nonexistent_template"},
        })
        assert resp.status_code == 422
        assert "does not exist" in resp.json()["detail"]

    def test_create_valid_provider_201(self, app_client, test_env):
        resp = app_client.post("/providers", json={
            "id": "dr_jones",
            "name": "Dr. Jones",
            "credentials": "DO",
            "specialty": "orthopedic",
            "template_routing": {"initial_evaluation": "ortho_initial_eval"},
            "style_directives": ["Use active voice"],
            "custom_vocabulary": ["arthroscopy"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "dr_jones"
        assert data["specialty"] == "orthopedic"

    def test_create_empty_specialty_allowed(self, app_client, test_env):
        """Empty specialty should bypass dictionary validation."""
        resp = app_client.post("/providers", json={
            "id": "dr_general",
            "name": "Dr. General",
            "specialty": "",
        })
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Provider update validation tests
# ---------------------------------------------------------------------------

class TestProviderUpdateValidation:
    def test_update_missing_specialty_dict_422(self, app_client):
        resp = app_client.put("/providers/dr_smith", json={
            "specialty": "dermatology",
        })
        assert resp.status_code == 422
        assert "no dictionary" in resp.json()["detail"].lower()

    def test_update_invalid_template_routing_422(self, app_client, test_env):
        """Updating template routing to nonexistent template should return 422.

        Note: providers.py update_provider() constructs Path("config/templates")
        locally. We verify that nonexistent_template doesn't exist at that path
        either (it shouldn't in test env or real env), so the validation fires.
        """
        resp = app_client.put("/providers/dr_smith", json={
            "template_routing": {"follow_up": "nonexistent_template"},
        })
        assert resp.status_code == 422
        assert "does not exist" in resp.json()["detail"]

    def test_update_nonexistent_provider_404(self, app_client):
        resp = app_client.put("/providers/nonexistent", json={"name": "Updated"})
        assert resp.status_code == 404

    def test_update_valid_fields(self, app_client, test_env):
        resp = app_client.put("/providers/dr_smith", json={
            "name": "Dr. Smith Updated",
            "credentials": "MD, FAAOS",
            "style_directives": ["Use passive voice", "Include ICD-10 codes"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Dr. Smith Updated"


# ---------------------------------------------------------------------------
# Specialty audit consistency tests
# ---------------------------------------------------------------------------

class TestSpecialtyAudit:
    def test_audit_clean_state(self, app_client):
        """No issues when everything is consistent."""
        resp = app_client.get("/specialties/audit/consistency")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_issues"] == 0

    def test_audit_detects_missing_specialty_dict(self, app_client, test_env):
        """Template referencing a specialty with no dictionary file."""
        # Create a template with a nonexistent specialty
        bad_tpl = {
            "name": "Bad Template",
            "specialty": "dermatology",
            "visit_type": "initial_evaluation",
        }
        with open(test_env / "config" / "templates" / "bad_tpl.yaml", "w") as f:
            yaml.dump(bad_tpl, f)

        resp = app_client.get("/specialties/audit/consistency")
        data = resp.json()
        assert data["errors"] >= 1
        assert any("dermatology" in i["message"] for i in data["issues"])

    def test_audit_detects_provider_missing_specialty(self, app_client, test_env):
        """Provider with specialty that has no dictionary."""
        bad_provider = {
            "id": "dr_bad",
            "name": "Dr. Bad",
            "specialty": "podiatry",
            "template_routing": {},
        }
        with open(test_env / "config" / "providers" / "dr_bad.yaml", "w") as f:
            yaml.dump(bad_provider, f)

        resp = app_client.get("/specialties/audit/consistency")
        data = resp.json()
        assert any("podiatry" in i["message"] for i in data["issues"])

    def test_audit_detects_broken_template_routing(self, app_client, test_env):
        """Provider routing to a nonexistent template."""
        bad_provider = {
            "id": "dr_broken",
            "name": "Dr. Broken",
            "specialty": "orthopedic",
            "template_routing": {"initial_evaluation": "nonexistent_template"},
        }
        with open(test_env / "config" / "providers" / "dr_broken.yaml", "w") as f:
            yaml.dump(bad_provider, f)

        resp = app_client.get("/specialties/audit/consistency")
        data = resp.json()
        assert any("nonexistent_template" in i["message"] for i in data["issues"])

    def test_audit_detects_specialty_mismatch(self, app_client, test_env):
        """Provider specialty doesn't match template specialty (warning)."""
        # Create a neurology template
        neuro_tpl = {
            "name": "Neuro Initial",
            "specialty": "neurology",
            "visit_type": "initial_evaluation",
        }
        with open(test_env / "config" / "templates" / "neuro_initial.yaml", "w") as f:
            yaml.dump(neuro_tpl, f)

        # Provider is orthopedic but routes to neuro template
        mismatch_provider = {
            "id": "dr_mismatch",
            "name": "Dr. Mismatch",
            "specialty": "orthopedic",
            "template_routing": {"initial_evaluation": "neuro_initial"},
        }
        with open(test_env / "config" / "providers" / "dr_mismatch.yaml", "w") as f:
            yaml.dump(mismatch_provider, f)

        resp = app_client.get("/specialties/audit/consistency")
        data = resp.json()
        assert data["warnings"] >= 1
        assert any("specialty mismatch" in i["message"].lower() for i in data["issues"])


# ---------------------------------------------------------------------------
# Specialty CRUD edge cases
# ---------------------------------------------------------------------------

class TestSpecialtyCRUD:
    def test_create_duplicate_specialty_409(self, app_client):
        resp = app_client.post("/specialties", json={"id": "orthopedic", "terms": ["test"]})
        assert resp.status_code == 409

    def test_get_nonexistent_specialty_404(self, app_client):
        resp = app_client.get("/specialties/nonexistent")
        assert resp.status_code == 404

    def test_update_nonexistent_specialty_404(self, app_client):
        resp = app_client.put("/specialties/nonexistent/dictionary", json={"terms": ["test"]})
        assert resp.status_code == 404

    def test_list_specialties_excludes_base_medical(self, app_client, test_env):
        """base_medical.txt should not appear as a specialty."""
        (test_env / "config" / "dictionaries" / "base_medical.txt").write_text("# base\nterm\n")
        resp = app_client.get("/specialties")
        ids = [s["id"] for s in resp.json()]
        assert "base_medical" not in ids
