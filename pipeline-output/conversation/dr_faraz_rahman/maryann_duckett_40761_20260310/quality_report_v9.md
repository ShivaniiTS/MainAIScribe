# Quality Report — maryann_duckett_40761_20260310

**Pipeline Version:** v9  
**Overall Score:** 4.5 / 5.0  
**Keyword Overlap:** 8%  
**Evaluation time:** 6.9s  

---

## Dimension Scores

| Dimension | Score | Weight | Assessment |
|-----------|-------|--------|------------|
| Medical Accuracy | 4.0/5 ★★★★☆ | 30% | ✓ Good |
| Completeness | 5.0/5 ★★★★★ | 25% | ✓ Good |
| No Hallucination | 5.0/5 ★★★★★ | 20% | ✓ Good |
| Structure Compliance | 5.0/5 ★★★★★ | 10% | ✓ Good |
| Clinical Language | 3.0/5 ★★★☆☆ | 10% | ⚠ Fair |
| Readability | 4.0/5 ★★★★☆ | 5% | ✓ Good |

**Evaluator notes:**
> The generated note is mostly accurate, but lacks some specific details from the gold standard. It correctly captures the patient's chief complaint and history of present illness, but omits some relevant information about past medical and surgical history. The clinical language is mostly formal, but with a few lapses in terminology.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 1 | {'name': 'tramadol', 'dose': '50 milligrams', 'frequency': 'as needed every four hours'} |
| Diagnoses | 0 | 2 | {'description': 'low back pain and a back contusion', 'laterality': 'right-hand side', 'etiology': 'auto accident on february 16th'}, {'description': 'degenerative changes in the lumbar spine', 'laterality': 'lumbar spine', 'etiology': ''} |
| Exam findings | 0 | 6 | {'finding': 'tender to palpation', 'location': 'neck and back'}, {'finding': 'limited flexion', 'measurement': '60 degrees', 'location': 'lumbar spine'}, {'finding': 'limited extension', 'measurement': '20 degrees', 'location': 'lumbar spine'}, {'finding': 'lateral tension increased pain', 'location': 'right and left directions'}, {'finding': 'tenderness along the anterior aspect', 'location': 'leg'} |
| Plan items | 0 | 1 | {'item': 'continue taking tramadol as needed every four hours'} |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, PAST SURGICAL HISTORY, SOCIAL HISTORY, FAMILY HISTORY, REVIEW OF SYSTEMS  
