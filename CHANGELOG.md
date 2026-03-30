# Changelog

## 2026-03-30 — Dual-Audio Upload Support

Merged `feature/dual-audio-upload` into `main`. Adds dual-audio upload for conversation mode — primary audio and note dictation audio (`note_audio.mp3`) can now be uploaded in a single multipart request. Includes MDCO integration support so the MD Checkout AI Scribe module can leverage dual-audio through the middleware. Also adds `.gitignore` entries for runtime data directories (`ai-scribe-data/`, `output/`, `pipeline-data/`, `pipeline-output/`).

Files modified:
- `.gitignore` — ignore runtime data directories
- `api/models.py` — added note_audio field to encounter model
- `api/pipeline/routes.py` — accept note_audio in pipeline upload
- `api/proxy.py` — forward note_audio through PHI-isolating proxy
- `api/routes/encounters.py` — dual-audio upload endpoint, audio serving with `?type=notes`
- `client/mobile/src/lib/api.ts` — mobile client dual-audio support
- `config/deployment.yaml` — config tweak
- `scripts/sync_to_pipeline.py` — minor cleanup
