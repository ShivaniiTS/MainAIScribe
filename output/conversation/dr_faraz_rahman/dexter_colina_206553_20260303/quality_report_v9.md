# Quality Report — dexter_colina_206553_20260303

**Pipeline Version:** v9  
**Overall Score:** 4.3 / 5.0  
**Keyword Overlap:** 28%  
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
> The generated note accurately captures the patient's symptoms, diagnosis, and treatment plan. However, it lacks some details from the transcript, such as the patient's past medical history and family history. The language is mostly clinical, but there are a few instances of informal terminology.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 1 | {'name': 'over-the-counter medication', 'dose': '', 'frequency': 'as needed'} |
| Diagnoses | 1 | 1 | — |
| Exam findings | 4 | 4 | — |
| Plan items | 1 | 1 | — |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, PAST SURGICAL HISTORY, SOCIAL HISTORY, FAMILY HISTORY, REVIEW OF SYSTEMS  
