# Quality Report — david_augustyniak_226974_20260219

**Pipeline Version:** v9  
**Overall Score:** 4.3 / 5.0  
**Keyword Overlap:** 26%  
**Evaluation time:** 9.8s  

---

## Dimension Scores

| Dimension | Score | Weight | Assessment |
|-----------|-------|--------|------------|
| Medical Accuracy | 4.0/5 ★★★★☆ | 30% | ✓ Good |
| Completeness | 4.0/5 ★★★★☆ | 25% | ✓ Good |
| No Hallucination | 5.0/5 ★★★★★ | 20% | ✓ Good |
| Structure Compliance | 5.0/5 ★★★★★ | 10% | ✓ Good |
| Clinical Language | 4.0/5 ★★★★☆ | 10% | ✓ Good |
| Readability | 4.0/5 ★★★★☆ | 5% | ✓ Good |

**Evaluator notes:**
> The generated note accurately captures the patient's symptoms, treatment plan, and medical history. However, it lacks some specific details mentioned in the gold standard note, such as the patient's past medical issues and surgical history.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 1 | {'name': 'anticoagulation', 'dose': '', 'frequency': ''} |
| Diagnoses | 4 | 4 | — |
| Exam findings | 4 | 4 | — |
| Plan items | 2 | 2 | — |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** HISTORY OF PRESENT ILLNESS, REVIEW OF SYSTEMS  
