# AI Scribe Enterprise — Development Collaboration Guide

This document defines the process for teams collaborating on the AI Scribe codebase. It covers repository setup, branching strategy, testing requirements, code review, and release management.

---

## 1. Repository Setup

### Clone the Repository

```bash
git clone https://github.com/sn-talisman/ai-scribe-enterprise.git
cd ai-scribe-enterprise
```

### Install Dependencies

```bash
# Python (backend + pipeline)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Node.js (web UI)
cd client/web && npm install && cd ../..

# Mobile app
cd client/mobile && npm install && cd ../..
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env and set:
#   HF_TOKEN=<your HuggingFace token>  (required for pyannote diarization)
source .env
```

### Verify Setup

```bash
# Run the test suite
source .venv/bin/activate
python -m pytest tests/unit/ tests/integration/ --tb=short

# Start both servers
AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --port 8000 &
AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --port 8100 &

# Verify
curl http://localhost:8000/health   # {"status":"ok","role":"provider-facing"}
curl http://localhost:8100/health   # {"status":"ok","role":"processing-pipeline"}
```

---

## 2. Branching Strategy

### Branch Naming Convention

```
<team>/<type>/<short-description>

Examples:
  provider/feature/patient-search-typeahead
  provider/fix/settings-race-condition
  pipeline/feature/batch-retry-logic
  pipeline/fix/vram-oom-on-large-audio
  shared/refactor/deployment-config-cleanup
  shared/docs/update-architecture
```

**Prefixes:**
- `provider/` — Provider-facing team (API, web UI, mobile, EHR, deployment)
- `pipeline/` — Pipeline team (ASR, LLM, quality, admin CRUD, batch processing)
- `shared/` — Changes that affect both teams (config, models, deployment, docs)

**Types:**
- `feature/` — New functionality
- `fix/` — Bug fix
- `refactor/` — Code restructuring without behavior change
- `docs/` — Documentation only
- `test/` — Test additions or improvements

### Creating a Branch

```bash
# Always branch from latest main
git checkout main
git pull origin main
git checkout -b provider/feature/my-feature
```

### Keeping Your Branch Up to Date

```bash
# Rebase onto main regularly (at least daily for active branches)
git fetch origin
git rebase origin/main

# If conflicts arise, resolve them and continue
git rebase --continue
```

---

## 3. Testing Requirements

### Test Tiers

| Tier | Location | What It Tests | When to Run |
|------|----------|---------------|-------------|
| **Unit** | `tests/unit/` | Individual functions, API contracts, config logic | Every commit |
| **Integration** | `tests/integration/` | Cross-server communication, role isolation, data flow | Every commit |
| **E2E** | `tests/e2e/` | Full workflow: upload → pipeline → output retrieval | Before merge to main |
| **Web page** | Shell scripts | All UI pages render without errors (200, no Turbopack crashes) | Before merge to main |
| **API sweep** | Shell scripts | All API endpoints return expected status codes + data | Before merge to main |

### What Each Test Must Verify

Tests must validate **data content**, not just HTTP status codes. A test that only checks `assert resp.status_code == 200` is incomplete. Always verify:

1. **Response shape** — Required fields are present
2. **Data population** — Fields that should have values are not null/empty
3. **Cross-server consistency** — Both servers return the same data for shared endpoints
4. **Error cases** — 404 for missing resources, 403 for unauthorized features

**Example — good test:**
```python
def test_encounters_have_quality_scores(self, client, mock_env):
    """Samples with a quality report must have populated scores."""
    _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v9"])
    _write_quality_report(output_dir, "v9")
    resp = client.get("/encounters")
    s1 = resp.json()[0]
    assert s1["quality"] is not None, "Must have quality when report exists"
    assert s1["quality"]["overall"] == 4.5
```

**Example — insufficient test (catches status but misses data bugs):**
```python
def test_encounters(self, client):
    resp = client.get("/encounters")
    assert resp.status_code == 200  # This passes even with empty quality!
```

### Running Tests

```bash
# Unit + integration (fast, run on every commit)
python -m pytest tests/unit/ tests/integration/ --tb=short -q

# Full suite including E2E (slower, run before merge)
python -m pytest tests/ --tb=short -v

# Web page + API sweep (run before merge)
# These scripts are in the repo root and test live servers
bash scripts/test_api_sweep.sh
bash scripts/test_web_pages.sh
```

### Writing New Tests

- **Provider team:** Add tests to `tests/unit/test_api_contracts.py` for new API endpoints, `tests/integration/test_api_encounters.py` or `test_api_providers.py` for provider-specific flows
- **Pipeline team:** Add tests to `tests/unit/test_pipeline_api.py` for pipeline routes, `tests/integration/test_dual_server.py` for cross-server behavior
- **Shared changes:** Add tests to `tests/unit/test_deployment_config.py` for config changes, `tests/unit/test_paths.py` for path resolution

---

## 4. Commit Standards

### Commit Message Format

```
<summary line — what changed and why, under 72 chars>

<optional body — details, context, trade-offs>

<optional footer — references, co-authors>
```

**Good examples:**
```
Fix quality scores missing on pipeline server

The pipeline server's OUTPUT_DIR (pipeline-output/) had generated notes
but no quality_report_v9.md. The data loader reads quality from this
aggregate file, so all scores were null on port 8100.

Fix: copy quality reports to pipeline-output/ during batch processing.
Add data integrity test to catch this class of bug.
```

```
Add patient search typeahead to Capture page

Queries GET /patients/search with debounced input. Shows name, DOB, MRN
in dropdown. Selected patient auto-populates demographics.
```

### Pre-Commit Checklist

Before every commit:

1. **Tests pass:** `python -m pytest tests/unit/ tests/integration/ --tb=short`
2. **No unintended files:** `git diff --cached --stat` — review what you're committing
3. **No secrets:** Never commit `.env`, API keys, tokens, or PHI
4. **No large files:** No `.mp3`, `node_modules/`, `.next/`, model weights
5. **No debug code:** Remove `console.log`, `print()` debugging, hardcoded test URLs

---

## 5. Code Review and Merge Process

### Pull Request Workflow

```
1. Push your branch
   git push origin provider/feature/my-feature

2. Create PR on GitHub
   gh pr create --title "Add patient search typeahead" \
     --body "## Summary\n- Added typeahead to Capture page\n\n## Test Plan\n- Unit tests for search endpoint\n- Manual test on mobile"

3. Assign reviewer from the other team for shared changes,
   or from your own team for role-specific changes

4. Address review comments, push updates

5. Merge after approval + CI green
```

### Review Criteria

Reviewers should check:

- [ ] Tests cover the change (data integrity, not just status codes)
- [ ] No regression in existing tests
- [ ] PHI isolation maintained (provider-facing changes don't leak to pipeline)
- [ ] Feature flags respected (admin features gated on processing-pipeline role)
- [ ] No hardcoded URLs, ports, or paths (use `config/deployment.yaml`)
- [ ] API changes are backward-compatible or documented as breaking

### Merge Requirements

- All unit + integration tests pass
- E2E tests pass for changes that touch the pipeline or API
- At least 1 approval from a team member
- No merge conflicts with main
- Squash merge for feature branches (clean history)

```bash
# Squash merge from GitHub UI, or:
git checkout main
git merge --squash provider/feature/my-feature
git commit  # Write a clean summary commit message
git push origin main
```

---

## 6. Release Tagging

### Tag Format

```
<milestone>-v<major>.<minor>

Examples:
  split-server-v2.1
  pipeline-v9
  mobile-v1.0
```

### When to Tag

- **Milestone completion** — A significant feature or phase is done
- **Before production deployment** — Tag the exact commit being deployed
- **After breaking changes** — Tag the new baseline

### Tagging Process

```bash
# 1. Ensure main is up to date and all tests pass
git checkout main
git pull origin main
python -m pytest tests/unit/ tests/integration/ --tb=short

# 2. Create annotated tag
git tag -a "split-server-v3" -m "Split Server v3: patient search, batch retry, mobile offline"

# 3. Push tag
git push origin --tags
```

### Critical Files to Update Before Tagging

Before creating a release tag, update these files:

| File | What to Update |
|------|----------------|
| `README.md` | Quality scores, sample counts, provider list, version table |
| `CLAUDE.md` | Session progress, version tracking table (v1 → current) |
| `docs/architecture.md` | Any architectural changes |
| `config/deployment.yaml` | Default role, sync settings if changed |
| `docs/dual_server_guide.md` | API route changes, new feature flags |
| `docs/team_responsibilities.md` | Team ownership changes |

```bash
# Verify these files are up to date
git diff HEAD -- README.md CLAUDE.md docs/architecture.md config/deployment.yaml

# If changes needed, commit them before tagging
git add README.md CLAUDE.md
git commit -m "Update docs for split-server-v3 release"
git tag -a "split-server-v3" -m "Description of this release"
git push origin main --tags
```

---

## 7. Handling Shared Code

Both teams share the same codebase. Changes to shared modules need coordination:

### Shared Modules (coordinate changes)

| Module | Impact |
|--------|--------|
| `config/deployment.py` | Server role logic, feature flags — affects both teams |
| `config/deployment.yaml` | Network config, sync settings — affects both teams |
| `config/paths.py` | Data directory resolution — affects both teams |
| `api/models.py` | API response schemas — frontend depends on these |
| `api/main.py` | Route mounting — determines which endpoints each server has |
| `api/data_loader.py` | Data reading logic — both servers use this |
| `orchestrator/state.py` | Pipeline state schema — pipeline team owns, provider team consumes |

### Rules for Shared Code

1. **Announce in team channel before modifying** — "I'm changing `api/models.py` to add `is_test` field"
2. **Add backward-compatible fields** — New optional fields with defaults, never remove fields
3. **Run both server roles' tests** — `python -m pytest tests/ --tb=short`
4. **Tag the commit** — Shared changes should be tagged so both teams can sync

---

## 8. Hotfix Process

For urgent production fixes:

```bash
# 1. Branch from the deployed tag
git checkout split-server-v2.1
git checkout -b hotfix/quality-scores-null

# 2. Fix, test, commit
# ... make changes ...
python -m pytest tests/unit/ tests/integration/ --tb=short
git commit -m "Fix null quality scores on pipeline server"

# 3. Merge to main
git checkout main
git merge hotfix/quality-scores-null
git push origin main

# 4. Tag the hotfix
git tag -a "split-server-v2.2" -m "Hotfix: quality scores on pipeline server"
git push origin --tags

# 5. Clean up
git branch -d hotfix/quality-scores-null
```

---

## 9. Quick Reference

```bash
# Daily workflow
git checkout main && git pull
git checkout -b provider/feature/my-work
# ... code ...
python -m pytest tests/unit/ tests/integration/ --tb=short
git add <files> && git commit -m "Description"
git push origin provider/feature/my-work
# Create PR on GitHub

# Before merge
python -m pytest tests/ --tb=short -v   # Full suite
# Verify both servers work:
AI_SCRIBE_SERVER_ROLE=provider-facing python -m pytest tests/ --tb=short
AI_SCRIBE_SERVER_ROLE=processing-pipeline python -m pytest tests/ --tb=short

# Release
git checkout main && git pull
git tag -a "milestone-vX.Y" -m "Release description"
git push origin main --tags
```
