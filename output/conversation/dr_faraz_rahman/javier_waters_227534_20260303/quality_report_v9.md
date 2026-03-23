# Quality Report — javier_waters_227534_20260303

**Pipeline Version:** v9  
**Overall Score:** 4.5 / 5.0  
**Keyword Overlap:** 6%  
**Evaluation time:** 6.8s  

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
> The generated note is mostly accurate, with only minor inaccuracies in the medical history section. However, it lacks some important details such as the patient's past surgical history and family history. The language used is mostly clinical, but there are some lapses in terminology. Overall, the structure of the note is correct, but it could be improved by including more relevant content.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 0 | — |
| Diagnoses | 0 | 2 | {'description': 'persistent pain in neck and back', 'laterality': 'left side', 'etiology': 'motor vehicle accident'}, {'description': 'post-traumatic headaches', 'etiology': 'motor vehicle accident'} |
| Exam findings | 0 | 5 | {'finding': 'paracervical tenderness', 'location': 'neck'}, {'finding': 'trapezius tenderness', 'location': 'neck'}, {'finding': 'limited range of motion', 'degrees': none, 'grades': none, 'location': 'cervical spine', 'directions': ['flexion extension', 'lateral rotation', 'lateral bending']}, {'finding': 'paraspinal tenderness', 'location': 'thoracic spine'}, {'finding': 'paraspinal tenderness', 'location': 'lumbar spine'} |
| Plan items | 0 | 2 | {'item': 'physical therapy treatment', 'start_date': 'tomorrow'}, {'item': 'x-rays of neck, back, and left femur', 'purpose': 'evaluate'} |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, PAST SURGICAL HISTORY, SOCIAL HISTORY, FAMILY HISTORY, REVIEW OF SYSTEMS  
