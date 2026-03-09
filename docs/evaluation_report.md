# Named Entity Recognition Evaluation Report: Archaeological Excavation Data

## 1. Overview

This report documents the performance of a fine-tuned GLiNER2 model applied to archaeological excavation reports. GLiNER2 is a transformer-based model architecture that utilizes bi-encoder matching for span-based entity recognition. The model was fine-tuned to extract eight domain-specific entity types from unstructured text.

## 2. Methodology

The evaluation was conducted using 5-fold cross-validation on a dataset consisting of 87 manually annotated records. The dataset was partitioned into five subsets; in each iteration, the model was trained on four subsets and validated on the fifth.

### 2.1 Configuration

- **Base Model**: GLiNER2 (Bi-encoder architecture)
- **Training Epochs**: 10 per fold
- **Evaluation Metric**: Strict span matching (IOB2-equivalent)
- **Labels**: ARTEFACT, PERIOD, LOCATION, CONTEXT, FEATURE, CONTEXT_ID, MATERIAL, SPECIES.

## 3. Results

### 3.1 Performance Comparison

| Metric        | Zero-Shot (Baseline) | Fine-Tuned (Average) |
| :------------ | :------------------- | :------------------- |
| **Precision** | 0.2932               | 0.7949               |
| **Recall**    | 0.4164               | 0.8764               |
| **F1 Score**  | 0.3441               | **0.8318**           |

### 3.2 Cross-Validation Metrics by Fold

| Fold        | Precision  | Recall     | F1 Score   |
| :---------- | :--------- | :--------- | :--------- |
| Fold 1      | 0.8276     | 0.8045     | 0.8159     |
| Fold 2      | 0.7277     | 0.9314     | 0.8170     |
| Fold 3      | 0.8509     | 0.9013     | 0.8754     |
| Fold 4      | 0.7867     | 0.8369     | 0.8110     |
| Fold 5      | 0.7815     | 0.9077     | 0.8399     |
| **Average** | **0.7949** | **0.8764** | **0.8318** |

## 4. Per-Label Analysis (Summary)

Performance varies by label based on the frequency of occurrences in the 87-record dataset.

- **High Precision (>0.90)**: `CONTEXT_ID` consistently scored near 0.96 F1.
- **Moderate Precision (0.75-0.85)**: `CONTEXT` and `ARTEFACT` labels maintained stable performance across all folds.
- **Variable/Low Support**: `LOCATION` performance fluctuated between 0.00 and 1.00 depending on the fold split. `FEATURE` and `MATERIAL` labels recorded 0.00 F1 scores, indicating insufficient training examples to establish statistical significance.

## 5. Summary

The fine-tuning process resulted in a significant increase in F1 score from 0.34 to 0.83. The model demonstrates high recall (0.87), making it suitable for metadata extraction tasks where minimizing false negatives is prioritized.
