#!/usr/bin/env python3
"""
Generate the AI Scribe Enterprise architecture diagram.
Output: docs/images/architecture_diagram.png

Layout (portrait):
  Left column  — LangGraph pipeline nodes stacked top-to-bottom
  Right column — MCP servers / support at the same rank as their calling node
  Bottom band  — API + Web UI

Run: source .venv/bin/activate && python3 scripts/generate_architecture_diagram.py
"""
from pathlib import Path
import graphviz

# ── palette ──────────────────────────────────────────────────────────────────
PIPE_FILL   = '#1E1B4B'
PIPE_DK     = '#312E81'
MCP_FILL    = '#00B27A'
EHR_FILL    = '#B45309'
WEB_FILL    = '#0369A1'
QUAL_FILL   = '#6D28D9'
FUTURE_FG   = '#475569'
FUTURE_FILL = '#E2E8F0'
WHITE       = 'white'


# ── helpers ──────────────────────────────────────────────────────────────────
def pipe_node(g, nid, title, sub='', w='2.0'):
    sub_part = f'<br/><font point-size="9">{sub}</font>' if sub else ''
    g.node(nid, f'<<b>{title}</b>{sub_part}>',
           shape='box', style='filled,rounded',
           fillcolor=PIPE_FILL, fontcolor=WHITE,
           fontname='Helvetica-Bold', fontsize='11',
           width=w, height='0.7', margin='0.12,0.08')


def router(g, nid, lbl):
    g.node(nid, f'<<font point-size="9">{lbl}</font>>',
           shape='diamond', style='filled',
           fillcolor=PIPE_DK, fontcolor=WHITE,
           fontsize='9', width='0.8', height='0.55')


def mcp_box(g, nid, title, lines, future=False, ehr=False, w='3.8'):
    body = ''.join(f'<br/><font point-size="9">{l}</font>' for l in lines)
    if future:
        fill, fg, sty = FUTURE_FILL, FUTURE_FG, 'filled,rounded,dashed'
        bdr = FUTURE_FG
    elif ehr:
        fill, fg, sty, bdr = EHR_FILL, WHITE, 'filled,rounded', EHR_FILL
    else:
        fill, fg, sty, bdr = MCP_FILL, WHITE, 'filled,rounded', MCP_FILL
    g.node(nid, f'<<b>{title}</b>{body}>',
           shape='box', style=sty,
           fillcolor=fill, fontcolor=fg, color=bdr,
           fontname='Helvetica-Bold', fontsize='10',
           width=w, margin='0.14,0.09')


def supp_box(g, nid, title, lines, accent=MCP_FILL, w='3.8'):
    body = ''.join(f'<br/><font point-size="9">{l}</font>' for l in lines)
    g.node(nid, f'<<b>{title}</b>{body}>',
           shape='box', style='filled,rounded',
           fillcolor='#F8FAFC', fontcolor='#1C1917', color=accent,
           fontname='Helvetica-Bold', fontsize='10',
           width=w, margin='0.14,0.09')


def web_box(g, nid, title, lines, w='5.5'):
    body = ''.join(f'<br/><font point-size="9">{l}</font>' for l in lines)
    g.node(nid, f'<<b>{title}</b>{body}>',
           shape='box', style='filled,rounded',
           fillcolor=WEB_FILL, fontcolor=WHITE,
           fontname='Helvetica-Bold', fontsize='10',
           width=w, margin='0.14,0.09')


# ── root graph ────────────────────────────────────────────────────────────────
d = graphviz.Digraph('ai_scribe', engine='dot')
d.attr(
    rankdir='TB',
    bgcolor=WHITE,
    compound='true',
    nodesep='0.40',
    ranksep='0.55',
    fontname='Helvetica',
    fontsize='12',
    dpi='150',
    pad='0.6',
)


# ════════════════════════════════════════════════════════════════════════════
# §1  LangGraph pipeline — vertical column on the left
#     MCP / support nodes placed rank='same' as their calling pipeline node
# ════════════════════════════════════════════════════════════════════════════

# ── CONTEXT ──
with d.subgraph() as r:
    r.attr(rank='same')
    pipe_node(r, 'ctx',  'CONTEXT',    'Load patient context')
    mcp_box(r, 'ehr_stub',
            'StubEHRServer  (mcp_servers/ehr/)',
            ['Reads patient_context.yaml per sample',
             'get_patient() · get_problem_list()',
             'get_medications() · get_allergies()',
             '[FUTURE] FHIRServer (R4) — same interface'],
            ehr=True)

# ── CAPTURE ──
with d.subgraph() as r:
    r.attr(rank='same')
    pipe_node(r, 'cap',  'CAPTURE',    'Audio I/O · VAD · noise suppress')
    supp_box(r, 'cap_tools',
             'Audio Tools  (mcp_servers/audio/)',
             ['Silero VAD — silence detection',
              'DeepFilterNet — noise suppression',
              'ffmpeg — format conversion',
              '[FUTURE] NeMo streaming ASR'],
             accent='#78716C')

# ── TRANSCRIBE ──
with d.subgraph() as r:
    r.attr(rank='same')
    pipe_node(r, 'tx',   'TRANSCRIBE', 'ASR · diarization · post-process')
    mcp_box(r, 'asr_wx',
            'WhisperXServer  (mcp_servers/asr/)  [DEFAULT]',
            ['faster-whisper · CTranslate2 · CUDA · VRAM ≈ 10–12 GB',
             'Model: whisper-large-v3 · beam_size=5',
             'pyannote 3.1 diarization · wav2vec2 word alignment',
             'condition_on_previous_text: True (dictation) / False (ambient)',
             'hotwords: top-100 provider vocab (logit boost)'])
    mcp_box(r, 'postproc',
            'PostProcessor  (postprocessor/)',
            ['medasr_postprocessor.py · 12-stage rule-based',
             '98 K OpenMedSpel dictionary (GPL)',
             'CTC stutter −90% · char stutter −99% · term fix −87%'],
            future=False, ehr=False, w='3.8')

# ── ASR router ──
with d.subgraph() as r:
    r.attr(rank='same')
    router(r, 'asr_r', 'asr_router')
    mcp_box(r, 'asr_lora',
            'WhisperXLoraServer  [OPT-IN]',
            ['peft LoRA (r=8) · use_lora=True in registry',
             'Ambient: −11–13 % WER · Dictation: NOT ready',
             'Requires ≥ 30 min verbatim audio per provider',
             'models/whisper_lora/{provider_id}/'],
            future=True)

# ── NOTE ──
with d.subgraph() as r:
    r.attr(rank='same')
    pipe_node(r, 'note', 'NOTE',       'Prompt assembly · LLM · parse')
    mcp_box(r, 'llm_ol',
            'OllamaServer  (mcp_servers/llm/)  [DEFAULT]',
            ['OpenAI-compat · localhost:11434/v1 · keep_alive=0',
             'qwen2.5:14b (Apache 2.0) · VRAM ≈ 8 GB',
             'Tasks: note_generation · coding · patient_summary',
             '[FUTURE] vLLM / Claude / OpenAI — same LLMEngine interface'])
    mcp_box(r, 'tpl_srv',
            'TemplateServer  (mcp_servers/data/)',
            ['config/templates/*.yaml · get_template(specialty, visit_type)',
             'ortho_follow_up (6 §) · ortho_initial_eval (12 §) · soap_default',
             'Routing: provider_id + visit_type → YAML → section prompts'],
            future=False, ehr=False, w='3.8')

# ── LLM router ──
with d.subgraph() as r:
    r.attr(rank='same')
    router(r, 'llm_r', 'llm_router')
    supp_box(r, 'prov_mgr',
             'Provider Profiles  (config/providers/)',
             ['{provider_id}.yaml · ProviderManager singleton',
              'style_directives · custom_vocabulary (22 terms)',
              'asr_overrides · template_routing · quality_history',
              'Dr. Faraz Rahman: 10 directives · 22 vocab · 4 routes'],
             accent=EHR_FILL)

# ── REVIEW ──
with d.subgraph() as r:
    r.attr(rank='same')
    pipe_node(r, 'rev',  'REVIEW',     'HITL review · auto-approve stub')
    supp_box(r, 'qual_fw',
             'Quality Framework  (quality/)',
             ['LLM-as-judge: always llama3.1:latest (cross-model fair eval)',
              'Dimensions: accuracy · completeness · no-hallucination',
              '            structure · clinical language  (each 1–5)',
              'fact_extractor: meds / diagnoses / exam findings precision/recall'],
             accent=QUAL_FILL)

# ── DELIVERY ──
with d.subgraph() as r:
    r.attr(rank='same')
    pipe_node(r, 'del',  'DELIVERY',   'EHR push · clipboard · output files')
    supp_box(r, 'output_files',
             'Output Files  (output/)',
             ['audio_transcript_v{N}.txt — standalone per version',
              'generated_note_v{N}.md — clinical note Markdown',
              'comparison_v{N}.md — gold vs generated side-by-side',
              'quality_report_v{N}.md · batch_report_v{N}.md'],
             accent=WEB_FILL)

# ── Pipeline edges ─────────────────────────────────────────────────────────
F9 = dict(fontsize='9', fontname='Helvetica')
d.edge('ctx',   'cap',   label='context_packet',    **F9)
d.edge('cap',   'tx',    label='audio_segments',     **F9)
d.edge('tx',    'asr_r',                             **F9)
d.edge('asr_r', 'note',  label='conf ≥ 0.40',        **F9)
d.edge('note',  'llm_r',                             **F9)
d.edge('llm_r', 'rev',   label='conf ≥ 0.50',        **F9)
d.edge('rev',   'del',   label='approved',            **F9)

# ── Call edges: pipeline → MCP (constraint=false → no rank interference) ──
CE = dict(style='dashed', fontsize='8', constraint='false', color='#94A3B8')
d.edge('ctx',  'ehr_stub',  **CE)
d.edge('cap',  'cap_tools', **CE)
d.edge('tx',   'asr_wx',    **CE)
d.edge('tx',   'postproc',  **CE)
d.edge('asr_r','asr_lora',  style='dashed', fontsize='8',
       constraint='false', color=FUTURE_FG)
d.edge('note', 'llm_ol',   **CE)
d.edge('note', 'tpl_srv',  **CE)
d.edge('llm_r','prov_mgr', **CE)
d.edge('rev',  'qual_fw',  **CE)
d.edge('del',  'output_files', **CE)


# ════════════════════════════════════════════════════════════════════════════
# EncounterState bar  — anchored below pipeline column
# ════════════════════════════════════════════════════════════════════════════
d.node('estate',
       '<<b>EncounterState (Pydantic v2)</b> — shared across all nodes   '
       '<font point-size="9">'
       'encounter_id · provider_id · recording_mode (AMBIENT | DICTATION) · '
       'context_packet · audio_segments · transcript · generated_note · '
       'final_note · delivery_result · errors[] · EncounterMetrics'
       '</font>>',
       shape='rectangle', style='filled',
       fillcolor='#C7D2FE', fontcolor=PIPE_FILL,
       fontsize='10', fontname='Helvetica',
       width='8.0', height='0.50')
d.edge('del', 'estate', style='invis')


# ════════════════════════════════════════════════════════════════════════════
# MCP Engine Registry (standalone — below pipeline, above API)
# ════════════════════════════════════════════════════════════════════════════
d.node('reg',
       '<<b>EngineRegistry</b>  (mcp_servers/registry.py)<br/>'
       '<font point-size="9">'
       'get_asr(provider_id, use_lora=False) · get_llm() · get_ehr() · '
       'get_with_failover() · health_check_all() · '
       'Config: config/engines.yaml — zero-code engine swap'
       '</font>>',
       shape='box', style='filled,rounded',
       fillcolor=MCP_FILL, fontcolor=WHITE,
       fontname='Helvetica-Bold', fontsize='11',
       width='9.0', margin='0.15,0.10')
d.edge('estate', 'reg', style='invis')


# ════════════════════════════════════════════════════════════════════════════
# Quality Progress table
# ════════════════════════════════════════════════════════════════════════════
d.node('qual_trend',
       '<<b>Quality Progress  (v1 → v6, judge: llama3.1:latest, 22 gold samples)</b><br/>'
       '<font point-size="9">'
       'v1  Session 4    basic end-to-end pipeline                     3.50 / 5.0    '
       'v2  Session 5    templates + specialty dictionaries             4.30 / 5.0<br/>'
       'v3  Session 7    EHR patient context + demographics             4.34 / 5.0    '
       'v4  Session 8    provider profiles + vocab + style directives   4.38 / 5.0<br/>'
       'v5  Session 10   ASR inference knobs + LoRA evaluation          4.35 / 5.0    '
       '<b>v6  Session 10c  qwen2.5:14b · two-pass eval                4.44 / 5.0  ✓</b>'
       '</font>>',
       shape='box', style='filled,rounded',
       fillcolor='#F1F5F9', fontcolor='#1C1917',
       fontname='Helvetica', fontsize='10',
       width='12.0', margin='0.15,0.10')
d.edge('reg', 'qual_trend', style='invis')


# ════════════════════════════════════════════════════════════════════════════
# §4  API + Web UI
# ════════════════════════════════════════════════════════════════════════════
with d.subgraph(name='cluster_ui') as u:
    u.attr(
        label='  API + WEB UI  ',
        labelloc='t', labeljust='l',
        style='filled', fillcolor='#EFF6FF',
        color=WEB_FILL, penwidth='2',
        fontname='Helvetica-Bold', fontsize='12',
    )

    with u.subgraph() as ui_row:
        ui_row.attr(rank='same')

        web_box(u, 'api',
                'FastAPI Backend  (api/)  ·  :8000',
                ['GET /encounters/{id}/transcript · /note · /audio · /comparison · /quality',
                 'POST /encounters · POST /{id}/upload  (triggers pipeline async)',
                 'GET /providers · GET /quality/aggregate · /samples · /dimensions',
                 'WS /ws/encounters/{id}  — real-time stage progress events',
                 'Data: output/ directory tree (no DB required)'])

        web_box(u, 'web',
                'Next.js Web App  (client/web/)  ·  :3000',
                ['Dashboard: KPI cards · quality trend (Recharts) · encounters table',
                 'Samples: filterable by version / mode / score range',
                 'Sample Detail  6 tabs:  Transcript (audio player + v-picker)',
                 '  Clinical Note (Markdown + v-picker) · Comparison (gold vs gen)',
                 '  Gold Standard · Quality Scores · Compare Versions (LCS diff)',
                 'Providers: quality history · Upload: drag-and-drop MP3/WAV'])

    u.edge('web', 'api', label='REST + WebSocket', fontsize='9')

d.edge('qual_trend', 'api', style='invis')   # anchor API below quality table


# ════════════════════════════════════════════════════════════════════════════
# Render
# ════════════════════════════════════════════════════════════════════════════
out_dir = Path('docs/images')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'architecture_diagram'

d.render(filename=str(out_path), format='png', cleanup=True)
print(f'✓  PNG : {out_path}.png')

dot_src = out_dir / 'architecture_diagram.dot'
dot_src.write_text(d.source)
print(f'✓  DOT : {dot_src}')
