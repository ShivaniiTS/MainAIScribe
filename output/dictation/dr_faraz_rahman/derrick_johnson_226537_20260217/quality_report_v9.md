# Quality Report — derrick_johnson_226537_20260217

**Pipeline Version:** v9  
**Overall Score:** 4.2 / 5.0  
**Keyword Overlap:** 6%  
**Evaluation time:** 6.6s  

---

## Dimension Scores

| Dimension | Score | Weight | Assessment |
|-----------|-------|--------|------------|
| Medical Accuracy | 4.0/5 ★★★★☆ | 30% | ✓ Good |
| Completeness | 5.0/5 ★★★★★ | 25% | ✓ Good |
| No Hallucination | 5.0/5 ★★★★★ | 20% | ✓ Good |
| Structure Compliance | 5.0/5 ★★★★★ | 10% | ✓ Good |
| Clinical Language | 1.0/5 ★☆☆☆☆ | 10% | ✗ Poor |
| Readability | 3.0/5 ★★★☆☆ | 5% | ⚠ Fair |

**Evaluator notes:**
> The generated note is mostly accurate, but lacks some specific details present in the gold standard. It correctly captures the patient's current symptoms and medical history, but uses very informal language, which negatively impacts its clinical language score.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 0 | — |
| Diagnoses | 0 | 2 | {'description': 'gunshot injury to left femur', 'laterality': 'left', 'etiology': 'gunshot'}, {'description': 'shifted rod in left femur', 'laterality': 'left', 'etiology': 'unknown'} |
| Exam findings | 0 | 5 | {'finding': 'headaches, intermittent', 'severity': 'out of 10', 'sensitivity': 'light sensitivity and forgetfulness'}, {'finding': 'neck pain', 'severity': 'out of 10', 'description': 'aversion, plunks, injuring head'}, {'finding': 'upper back pain', 'severity': 'out of 10', 'description': 'aversion, bending, twisting, and lifting'}, {'finding': 'low back pain', 'severity': 'out of 10', 'description': 'aversion, bending, twisting, and lifting'}, {'finding': 'left thigh pain', 'severity': 'out of 10', 'description': 'worse with direct pressure, bearing weight for extended periods, or climbing stairs'} |
| Plan items | 0 | 1 | further evaluation |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** HISTORY OF PRESENT ILLNESS, REVIEW OF SYSTEMS, PAST MEDICAL HISTORY  
