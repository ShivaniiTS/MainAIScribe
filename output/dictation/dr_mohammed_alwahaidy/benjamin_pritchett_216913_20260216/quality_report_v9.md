# Quality Report — benjamin_pritchett_216913_20260216

**Pipeline Version:** v9  
**Overall Score:** 4.1 / 5.0  
**Keyword Overlap:** 38%  
**Evaluation time:** 11.7s  

---

## Dimension Scores

| Dimension | Score | Weight | Assessment |
|-----------|-------|--------|------------|
| Medical Accuracy | 4.0/5 ★★★★☆ | 30% | ✓ Good |
| Completeness | 4.0/5 ★★★★☆ | 25% | ✓ Good |
| No Hallucination | 4.0/5 ★★★★☆ | 20% | ✓ Good |
| Structure Compliance | 5.0/5 ★★★★★ | 10% | ✓ Good |
| Clinical Language | 4.0/5 ★★★★☆ | 10% | ✓ Good |
| Readability | 4.0/5 ★★★★☆ | 5% | ✓ Good |

**Evaluator notes:**
> The generated note accurately captures the patient's medical history, physical examination findings, and treatment plan. However, it lacks some details from the gold standard note, such as the patient's prior history of depression and anxiety. Additionally, the generated note does not include the emergency room report review mentioned in the gold standard note.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 1 | {'name': 'anticoagulants', 'dose': '', 'frequency': ''} |
| Diagnoses | 9 | 9 | — |
| Exam findings | 5 | 5 | — |
| Plan items | 4 | 4 | — |

---

## Section Coverage

**Present:** HISTORY OF PRESENT ILLNESS, PHYSICAL EXAMINATION, ASSESSMENT, PLAN  
**Missing (vs gold):** PAST MEDICAL HISTORY  
