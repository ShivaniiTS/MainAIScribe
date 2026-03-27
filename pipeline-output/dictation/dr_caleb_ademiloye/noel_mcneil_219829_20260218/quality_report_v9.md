# Quality Report — noel_mcneil_219829_20260218

**Pipeline Version:** v9  
**Overall Score:** 4.5 / 5.0  
**Keyword Overlap:** 10%  
**Evaluation time:** 6.4s  

---

## Dimension Scores

| Dimension | Score | Weight | Assessment |
|-----------|-------|--------|------------|
| Medical Accuracy | 4.0/5 ★★★★☆ | 30% | ✓ Good |
| Completeness | 5.0/5 ★★★★★ | 25% | ✓ Good |
| No Hallucination | 5.0/5 ★★★★★ | 20% | ✓ Good |
| Structure Compliance | 5.0/5 ★★★★★ | 10% | ✓ Good |
| Clinical Language | 4.0/5 ★★★★☆ | 10% | ✓ Good |
| Readability | 3.0/5 ★★★☆☆ | 5% | ⚠ Fair |

**Evaluator notes:**
> The generated note is mostly accurate, with only minor inaccuracies in the patient's information. However, it lacks some important details such as the patient's past medical history and physical examination findings. The language used is mostly clinical, but there are some lapses in terminology. Overall, the structure of the note is correct, but it could be improved by including more relevant content.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 2 | {'name': 'lumbar epidural serum injection', 'dose': '', 'frequency': ''}, {'name': 'lumbar radiofrequency ablation', 'dose': '', 'frequency': ''} |
| Diagnoses | 0 | 2 | {'description': 'grade 1 spondylolisthesis, l5 on s1 with bilateral pars interarticularis defects'}, {'description': 'low back pain', 'laterality': 'axial and constant', 'etiology': ''} |
| Exam findings | 0 | 4 | {'finding': 'head is normal, spastic'}, {'finding': 'intramuscular extremities are negative for signs of edema'}, {'finding': 'proof of bolus are intact'}, {'finding': 'skin is intact'} |
| Plan items | 0 | 2 | {'item': 'continue with therapies here at this clinic'}, {'item': 'do not wish to have any more spine injections at this time'} |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, REVIEW OF SYSTEMS  
