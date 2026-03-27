"""
tests/integration/test_api_providers.py — Integration tests for provider API.

Tests provider listing, detail retrieval, and quality trend endpoints.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_config():
    with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "provider-facing"}):
        from config.deployment import get_deployment_config
        get_deployment_config(reload=True)
        yield
    from config.deployment import get_deployment_config
    get_deployment_config(reload=True)


@pytest.fixture
def client():
    from config.deployment import get_deployment_config
    get_deployment_config(reload=True)
    import importlib
    import api.main
    importlib.reload(api.main)
    return TestClient(api.main.app)


class TestProviderList:
    def test_list_returns_list(self, client):
        resp = client.get("/providers")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_provider_has_fields(self, client):
        resp = client.get("/providers")
        providers = resp.json()
        if not providers:
            pytest.skip("No providers available")
        p = providers[0]
        assert "id" in p
        assert "name" in p


class TestProviderDetail:
    def test_detail_not_found(self, client):
        resp = client.get("/providers/nonexistent_provider_xyz")
        # May return 404 or a default profile
        assert resp.status_code in (200, 404)

    def test_detail_existing_provider(self, client):
        list_resp = client.get("/providers")
        providers = list_resp.json()
        if not providers:
            pytest.skip("No providers available")

        pid = providers[0]["id"]
        resp = client.get(f"/providers/{pid}")
        assert resp.status_code == 200


class TestProviderQualityTrend:
    def test_quality_trend(self, client):
        list_resp = client.get("/providers")
        providers = list_resp.json()
        if not providers:
            pytest.skip("No providers available")

        pid = providers[0]["id"]
        resp = client.get(f"/providers/{pid}/quality-trend")
        assert resp.status_code == 200


class TestPatientSearch:
    def test_search_returns_list(self, client):
        resp = client.get("/patients/search")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_search_with_query(self, client):
        resp = client.get("/patients/search", params={"q": "test"})
        assert resp.status_code == 200


class TestSpecialtiesCRUD:
    def test_list_specialties(self, client):
        resp = client.get("/specialties")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_specialty_detail(self, client):
        list_resp = client.get("/specialties")
        specialties = list_resp.json()
        if not specialties:
            pytest.skip("No specialties available")
        sid = specialties[0]["id"]
        resp = client.get(f"/specialties/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert "terms" in data


class TestTemplatesCRUD:
    def test_list_templates(self, client):
        resp = client.get("/templates")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_template_detail(self, client):
        list_resp = client.get("/templates")
        templates = list_resp.json()
        if not templates:
            pytest.skip("No templates available")
        tid = templates[0]["id"]
        resp = client.get(f"/templates/{tid}")
        assert resp.status_code == 200
