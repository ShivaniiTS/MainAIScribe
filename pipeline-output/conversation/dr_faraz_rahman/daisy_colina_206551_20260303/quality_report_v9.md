# Quality Report — daisy_colina_206551_20260303

**Pipeline Version:** v9  
**Overall Score:** 3.5 / 5.0  
**Keyword Overlap:** 6%  
**Evaluation time:** 8.0s  

---

## Dimension Scores

| Dimension | Score | Weight | Assessment |
|-----------|-------|--------|------------|
| Medical Accuracy | 4.0/5 ★★★★☆ | 30% | ✓ Good |
| Completeness | 1.0/5 ★☆☆☆☆ | 25% | ✗ Poor |
| No Hallucination | 5.0/5 ★★★★★ | 20% | ✓ Good |
| Structure Compliance | 5.0/5 ★★★★★ | 10% | ✓ Good |
| Clinical Language | 4.0/5 ★★★★☆ | 10% | ✓ Good |
| Readability | 3.0/5 ★★★☆☆ | 5% | ⚠ Fair |

**Evaluator notes:**
> The generated note is mostly accurate, but lacks crucial information about the patient's history and current medications. It perfectly grounds itself in the transcript without any significant hallucinations.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 2 | {'name': 'cyclobenzaprine', 'dose': '10mg', 'frequency': 'qhs prn'}, {'name': 'naproxen', 'dose': '5-minimums', 'frequency': 'bid prn'} |
| Diagnoses | 0 | 3 | {'description': 'acute muscular ligamentous sprains, strain of cervical and thoracic spine, post-traumatic', 'laterality': '', 'etiology': 'post-traumatic'}, {'description': 'acute muscular ligamentous sprains, strain of lumbar spine, post-traumatic', 'laterality': '', 'etiology': 'post-traumatic'}, {'description': 'bilateral shoulder sprains, post-traumatic', 'laterality': 'bilateral', 'etiology': 'post-traumatic'} |
| Exam findings | 0 | 10 | {'finding': 'neck muscles tight', 'location': ''}, {'finding': 'paracervical tenderness', 'location': ''}, {'finding': 'trapezius tenderness', 'location': ''}, {'finding': 'limitation of flexion and extension', 'location': '', 'measurement': ''}, {'finding': 'cervical rotation uncomfortable', 'location': '', 'measurement': ''} |
| Plan items | 0 | 1 | {'item': 'x-rays of the cervical spine, thoracic spine, and lumbar spine to evaluate any bony abnormalities'} |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, PAST SURGICAL HISTORY, SOCIAL HISTORY, FAMILY HISTORY, REVIEW OF SYSTEMS  
