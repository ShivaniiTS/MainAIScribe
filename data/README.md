# Test Data

This folder previously held test data directly. All encounter data has been moved to `ai-scribe-data/` at the project root.

## Download Test Dataset

Audio files for testing are available on SharePoint:

**[Download Test Dataset](https://talismansolutionscom.sharepoint.com/:f:/s/ExcelsiaITprojects-AIScribe/IgBCWV3umPwaRowR3nC8dGVGAV5_VlQBAOBqJuDOewniM58?e=s9fYUE)**

The repository includes encounter metadata (patient demographics, gold-standard notes,
encounter details) but **excludes audio files** (`.mp3`) due to size and PHI concerns.
After downloading, place the audio files into their corresponding `ai-scribe-data/`
encounter folders — the folder names in the SharePoint archive match the repo structure.

## Data Directory Layout

```
ai-scribe-data/
├── conversation/                           # Ambient (multi-speaker) encounters
│   └── <physician_id>/
│       └── <patient>_<mrn>_<date>/
│           ├── conversation_audio.mp3      # Doctor-patient conversation (from SharePoint)
│           ├── note_audio.mp3              # Optional: physician dictation after visit
│           ├── final_soap_note.md          # Gold-standard clinical note
│           ├── patient_demographics.json   # Patient name, DOB, sex, MRN
│           └── encounter_details.json      # Visit type, provider, date, mode
│
├── dictation/                              # Single-speaker physician dictations
│   └── <physician_id>/
│       └── <patient>_<mrn>_<date>/
│           ├── dictation.mp3               # Physician dictation audio (from SharePoint)
│           ├── final_soap_note.md          # Gold-standard clinical note
│           ├── patient_demographics.json
│           └── encounter_details.json
```

## Pipeline Output

Generated outputs are written to `output/` at the project root, mirroring the data structure:

```
output/
├── dictation/
│   └── <physician_id>/
│       └── <patient>_<mrn>_<date>/
│           ├── generated_note_v7.md        # AI-generated clinical note (versioned)
│           ├── audio_transcript_v7.txt     # Standalone transcript (versioned)
│           └── quality_report.md           # Quality evaluation (if gold note exists)
├── conversation/
│   └── (same structure)
├── batch_report_v7.md                      # Aggregate quality report per version
└── transcript_cache_v7.json                # Cached transcripts (gitignored)
```

## Configurable Paths

All data paths can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_SCRIBE_ROOT` | Auto-detected project root | Base project root |
| `AI_SCRIBE_DATA_DIR` | `{root}/ai-scribe-data` | Patient encounter data |
| `AI_SCRIBE_OUTPUT_DIR` | `{root}/output` | Pipeline output |
| `AI_SCRIBE_CONFIG_DIR` | `{root}/config` | Config files |

Set these in your `.env` file or export them before running the pipeline.
