# Quality Report — anthony_frye_226186_20260218

**Pipeline Version:** v9  
**Overall Score:** 4.2 / 5.0  
**Keyword Overlap:** 6%  
**Evaluation time:** 4.8s  

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
> The generated note is mostly accurate, but lacks some specific details present in the gold standard. It correctly captures the patient's complaints and medical history, but fails to include relevant information about the patient's MRI results and treatment plan. The clinical language is informal and lay terms are used, making it difficult to read.

---

## Fact Check

| Category | Found | Total | Missed |
|----------|-------|-------|--------|
| Medications | 0 | 0 | — |
| Diagnoses | 0 | 2 | {'description': 'medial meniscus tear', 'laterality': 'left', 'etiology': 'pedestrian-versus-motor vehicle accident'}, {'description': 'left knee joint effusion and baker cyst', 'laterality': 'left'} |
| Exam findings | 0 | 0 | — |
| Plan items | 0 | 3 | continue ongoing outpatient chiropractic and rehabilitative program, dme dispensed today, orthopedic evaluation scheduled with dr. pace on 5-27-2026 |

---

## Section Coverage

**Present:** INTERVAL HISTORY, CURRENT MEDICATIONS, PHYSICAL EXAMINATION, IMAGING / DIAGNOSTICS, ASSESSMENT, PLAN  
**Missing (vs gold):** HISTORY OF PRESENT ILLNESS, CHIEF COMPLAINT, PAST MEDICAL HISTORY, REVIEW OF SYSTEMS  
