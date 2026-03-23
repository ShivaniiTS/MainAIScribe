# Quality Report — james_dotson_224938_20260224

**Pipeline Version:** v9  
**Overall Score:** 4.5 / 5.0  
**Keyword Overlap:** 6%  
**Evaluation time:** 6.0s  

---

## Dimension Scores

| Dimension | Score | Weight | Assessment |
|-----------|-------|--------|------------|
| Medical Accuracy | 4.0/5 ★★★★☆ | 30% | ✓ Good |
| Completeness | 5.0/5 ★★★★★ | 25% | ✓ Good |
| No Hallucination | 5.0/5 ★★★★★ | 20% | ✓ Good |
| Structure Compliance | 5.0/5 ★★★★★ | 10% | ✓ Good |
| Clinical Language | 4.0/5 ★★★★☆ | 10% | ✓ Good |
| Readability | 4.0/5 ★★★★☆ | 5% | ✓ Good |

**Evaluator notes:**
> The generated note is mostly accurate, with only minor inaccuracies in the medical history section. However, it lacks some important details such as the patient's occupation and living situation. The structure of the note is perfect, but the clinical language could be improved to match the gold standard.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 0 | — |
| Diagnoses | 0 | 2 | {'description': 'persistent pain in multiple areas', 'laterality': ['low back', 'right shoulder', 'left wrist', 'right knee'], 'etiology': []}, {'description': 'tenderness upon palpation', 'laterality': ['lumbar spine', 'right shoulder', 'right knee'], 'etiology': []} |
| Exam findings | 0 | 5 | {'finding': 'limited flexion to 30 degrees', 'location': 'lumbar spine'}, {'finding': 'tenderness in the lumbar spine', 'location': 'lumbar spine'}, {'finding': 'tender on the ac joint and lumbar area when pressed', 'location': 'right shoulder, lumbar area'}, {'finding': 'arm could be brought up to 80 degrees with abduction', 'location': 'right shoulder'}, {'finding': 'flexed forward to 90 degrees', 'location': 'right shoulder'} |
| Plan items | 0 | 1 | {'item': 'mri scans of the right shoulder, right knee, and low back are pending on march 16th'} |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, PAST SURGICAL HISTORY, SOCIAL HISTORY, FAMILY HISTORY, REVIEW OF SYSTEMS  
