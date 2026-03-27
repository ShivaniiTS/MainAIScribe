# Quality Report — antamar_corprew_225470_20260303

**Pipeline Version:** v9  
**Overall Score:** 4.5 / 5.0  
**Keyword Overlap:** 26%  
**Evaluation time:** 8.7s  

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
> The generated note accurately captures the patient's symptoms and treatment plan, but lacks some details present in the gold standard note. The language is mostly clinical, but there are a few instances of informal terminology.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 1 | {'name': 'over-the-counter medication', 'dose': '', 'frequency': ''} |
| Diagnoses | 3 | 3 | — |
| Exam findings | 3 | 3 | — |
| Plan items | 4 | 4 | — |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, PAST SURGICAL HISTORY, SOCIAL HISTORY, FAMILY HISTORY, REVIEW OF SYSTEMS  
