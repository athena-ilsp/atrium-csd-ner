# Project Strategy & Results Log

## Strategy

Our goal is to fine-tune a GLiNER2 model for specialized Named Entity Recognition on archaeological reports.

## Results Summary

### 1. Zero-Shot Baseline (Unsupervised)

- **Overall F1**: 0.3441

### 2. Fine-Tuned Results (5-Fold Cross Validation Summary)

The model was evaluated using 5-fold cross-validation on 87 manually annotated records.

- **Fold 1 F1**: 0.8159
- **Fold 2 F1**: 0.8170
- **Fold 3 F1**: 0.8754
- **Fold 4 F1**: 0.8110
- **Fold 5 F1**: 0.8399
- **Combined Average F1**: **0.8318**

**Analysis**: Fine-tuning produced a **142% improvement** in F1 score over the zero-shot baseline. The model is highly reliable for identifying `CONTEXT_ID`, `CONTEXT`, and `ARTEFACT`. Some labels (`FEATURE`, `LOCATION`) show variable performance across folds, likely due to low sample support in the 87 records.

## Project Reorganization

- [x] Consolidate Assets: Moved images and metadata inside `src/atrium_csd_ner/static/`.
- [x] Streamline Root: Removed boilerplate files (`tox.ini`, `Makefile`, etc.).
- [x] Infrastructure: Organized Docker and Deployment logic into `deployment/`.
- [x] Source Integration: Migrated to a professional package structure.

## Roadmap

- [x] Implement Argilla data extraction.
- [x] Evaluate GLiNER2 unsupervised performance.
- [x] Implement 5-fold cross-validation.
- [x] Complete all folds and evaluate final metrics lift.
- [x] Generate formal NER Evaluation Report.
- [x] Train final production model on 100% of data.
- [x] Deployment Infrastructure prepared for launch.
