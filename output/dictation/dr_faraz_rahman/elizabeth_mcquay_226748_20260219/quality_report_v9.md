# Quality Report — elizabeth_mcquay_226748_20260219

**Pipeline Version:** v9  
**Overall Score:** 4.5 / 5.0  
**Keyword Overlap:** 8%  
**Evaluation time:** 4.7s  

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
> The generated note is mostly accurate, but lacks some specific details present in the gold standard. It correctly captures the patient's history and assessment, but omits some relevant information such as past medical history and physical examination findings.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 1 | {'name': 'naproxen', 'dose': '500mg', 'frequency': 'sbid'} |
| Diagnoses | 0 | 0 | — |
| Exam findings | 0 | 6 | pain in neck (4-5/10), achiness and stiffness in the neck, worsening with laying down, long positions, and turning her head, pain in right wrist and hand (4-5/10), aching pains around the wrist and primarily into the thumb |
| Plan items | 0 | 0 | — |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, REVIEW OF SYSTEMS  
