# Quality Report — derrick_clarke_227512_20260303

**Pipeline Version:** v9  
**Overall Score:** 4.5 / 5.0  
**Keyword Overlap:** 4%  
**Evaluation time:** 5.7s  

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
> The generated note is mostly accurate, but lacks some specific details found in the gold standard. It correctly captures the patient's symptoms and medical history, but omits some relevant information such as the date of accident and the patient's past surgical history.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 2 | {'name': 'cyclobenzaprine', 'dose': '', 'frequency': ''}, {'name': 'hydrochlorothiazide', 'dose': '', 'frequency': ''} |
| Diagnoses | 0 | 2 | {'description': 'acute muscle metastatic sprain, strain of cervical and thoracic spine, post-traumatic'}, {'description': 'bilateral shoulder sprains, post-traumatic'} |
| Exam findings | 0 | 3 | {'finding': 'limited flexion', 'location': 'cervical spine', 'degrees': '90 left, 80 right'}, {'finding': 'pain with external rotation', 'location': 'shoulder blade', 'laterality': 'left more than right'}, {'finding': 'lumbar tenderness and spasm', 'location': 'lower back'} |
| Plan items | 0 | 1 | {'item': 'x-rays of the cervical spine, thoracic spine, and lumbar spine to evaluate for new bone abnormalities'} |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, PAST SURGICAL HISTORY, SOCIAL HISTORY, FAMILY HISTORY, REVIEW OF SYSTEMS  
