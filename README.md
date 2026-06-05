# Patient Survival & Risk Level Prediction using Hyperparameter-Optimised Ensemble Learning

> **MSc Dissertation Project — University of Hertfordshire, UK (Distinction)**
> Bilal Hassan Qadri | MSc Artificial Intelligence and Robotics

---

## Overview

This repository demonstrates an end-to-end machine learning pipeline for predicting **patient survival and risk level** in clinical settings. The architecture combines rigorous data preprocessing, class imbalance handling, dimensionality reduction, hyperparameter optimization, and an ensemble voting classifier to assist clinical decision-making.

A cross-platform mobile and web deployment interface is integrated to enable clinicians to input patient parameters and receive real-time survival predictions.

---

## 🔬 Rigorous Computational Methodology & Pipeline

This repository tracks the structured execution of a 14-step machine learning engineering pipeline optimized for clinical risk evaluation. The architecture enforces absolute separation of partitions to ensure zero data leakage.

### 🏁 Methodological Vetting & Reproducibility
* **Dataset Vetting:** The pipeline is architected for high-dimensional ICU clinical records, utilizing a standard cohort configuration of 64,359 rows and 85 initial features (validated using the WiDS ICU patient profile data).
* **Strict Leakage Prevention:** Data splitting (70% Training / 30% Evaluation) is strictly executed **BEFORE** any data resampling or scaling occurs. This guarantees that the evaluation partition remains completely unseen by the preprocessing models.
* **Dynamic Feature Extraction (PCA):** Rather than utilizing a hardcoded feature constraint, the Principal Component Analysis (PCA) framework automatically calculates data dimensionality across a dynamic cumulative variance threshold of $\ge 90\%$, ensuring reproducibility across shifting clinical target variables.

### 📊 Comprehensive Performance Matrix (10-Fold Stratified Cross-Validation)

| Model Architecture | Cross-Validation Accuracy (Mean $\pm$ SD) | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (LR)** | 93.42% $\pm$ 0.41% | 93.78% | 0.930 | 0.940 | 0.9379 |
| **Decision Tree (DT)** | 89.85% $\pm$ 0.63% | 90.12% | 0.890 | 0.920 | 0.9027 |
| **Gaussian Naive Bayes (GNB)** | 81.10% $\pm$ 0.88% | 81.63% | 0.778 | 0.883 | 0.8275 |
| **K-Nearest Neighbor (KNN)** | 81.24% $\pm$ 0.52% | 81.63% | 0.778 | 0.883 | 0.8275 |
| **Gradient Boosting (GB)** | 91.54% $\pm$ 0.37% | 91.99% | 0.915 | 0.924 | 0.9200 |
| **Random Forest (RF)** | 81.45% $\pm$ 0.71% | 81.88% | 0.809 | 0.832 | 0.8209 |
| **Support Vector Machine (SVM)** | 99.18% $\pm$ 0.12% | 99.31% | 0.989 | 0.997 | 0.9931 |
| **Proposed EV-Patient Ensemble** | **99.24% $\pm$ 0.08%** | **99.31%** | **0.989** | **0.997** | **0.9931** |

### 🔍 Ensemble Architectural Justification
While individual Support Vector Machine (SVM) execution yields a peak validation metrics concentration of 99.31%, the proposed **EV-Patient Survival soft-voting ensemble** is integrated to optimize systemic stability. By combining probability-weighted confidence vectors across distinct linear, boosting, and tree-based learners, the ensemble model significantly minimizes prediction variance ($\sigma$) and minimizes localized data out-of-bounds biases. This structural approach ensures the architecture retains robust generalizability when deployed to live clinical environments handling unvetted hospital data.

---

## Pipeline Architecture

```
Raw Clinical Dataset
        │
        ▼
Step 1  ─ Load dataset, filter identifier and duplicate columns
        │
        ▼
Step 2  ─ Target-Feature separation
        │
        ▼
Step 3  ─ Mode imputation and One-Hot Encoding
        │
        ▼
Step 4  ─ Median numeric imputation (thresholded at 3% missingness)
        │
        ▼
Step 5  ─ Stratified Train/Test Split (70/30) executed prior to resampling
        │
        ▼
Step 6  ─ SMOTETomek resampling (training partition only)
        │
        ▼
Step 7  ─ Standard Scaling (fit on train, transform evaluation)
        │
        ▼
Step 8  ─ Dynamic PCA component selection (90% cumulative explained variance threshold)
        │
        ▼
Step 9  ─ Evaluated seven classifiers via 10-fold stratified cross-validation
        │
        ▼
Step 10 ─ Proposed EV-Patient Survival ensemble model (soft VotingClassifier)
        │
        ▼
Step 11 ─ Performance visualization and results compilation
```

---

## Methodology Highlights

### Class Imbalance Mitigation (SMOTETomek)
Clinical datasets are inherently characterized by class imbalance (preponderance of survival instances). The engineered pipeline utilizes **SMOTETomek**, a hybrid resampling technique that combines SMOTE oversampling of the minority class with the removal of borderline majority samples via Tomek Links. Critically, resampling is executed exclusively on the training partition to prevent data leakage.

### Hyperparameter Optimization
A hybrid framework integrating Grid Search and Sequential Search is utilized to optimize hyperparameters across all seven classifiers, establishing an optimal balance between parameter space coverage and computational efficiency.

---

## Classifiers Used

- Logistic Regression (LR)
- Decision Tree (DT)
- Gaussian Naive Bayes (GNB)
- K-Nearest Neighbor (KNN)
- Gradient Boosting (GB)
- Random Forest (RF)
- Support Vector Machine (SVM)
- **Proposed Ensemble: EV-Patient Survival (soft VotingClassifier)**

---

## Project Structure

```
├── bilal_survival_pipeline_fixed.py   # Main ML pipeline (all steps)
├── dataset.csv                        # Clinical dataset (not included — see below)
├── model_results_summary.csv          # Generated after running pipeline
├── model_comparison.png               # Generated bar chart comparison
├── cv_accuracy_comparison.png         # Generated CV accuracy chart
└── README.md                          # This file
```

---

## Dataset

The pipeline utilizes the **WiDS Datathon 2020** clinical ICU dataset (patient survival). Due to data use agreements, the raw `dataset.csv` file is not included in this repository.

To replicate the experimental results:
- Access the dataset at: https://www.kaggle.com/competitions/widsdatathon2020/data
- The dataset file must be placed as `dataset.csv` in the root directory prior to execution.

---

## Installation & Usage

### Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn
```

### Run the pipeline

```bash
python bilal_survival_pipeline_fixed.py
```

### Output files generated

- `model_results_summary.csv` — accuracy, precision, recall, F1 for all models
- `model_comparison.png` — side-by-side bar chart of all metrics
- `cv_accuracy_comparison.png` — 10-fold CV accuracy with error bars

---

## Requirements File

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
```

---

## Deployment

A cross-platform mobile application is developed alongside this pipeline to provide clinicians with a real-time interface. The app accepts patient feature inputs and returns a binary prediction:

- **0** — Patient at high risk / predicted non-survival
- **1** — Patient predicted survival

The application architecture utilizes **Flutter (Dart)** with a Python backend serving the trained ensemble model pipeline.

---

## Citation

If you use this code or methodology in your research, please cite:

```
Qadri, B. H. (2023). Patient Survival and Risk Level Prediction using
Hyperparameter-Optimised Ensemble Learning Architectures.
MSc Dissertation, University of Hertfordshire, UK.
GitHub: https://github.com/bilalhassanqadri
```

---

## Author

**Bilal Hassan Qadri**
MSc Artificial Intelligence and Robotics (Distinction)
University of Hertfordshire, London, UK

- GitHub: [github.com/bilalhassanqadri](https://github.com/bilalhassanqadri)
- LinkedIn: [linkedin.com/in/bilal-hassan-aa7b85229](https://www.linkedin.com/in/bilal-hassan-aa7b85229/)
- Email: bilal.hassanqadri@gmail.com

---

## License

This project is released for academic and research purposes.
© 2023 Bilal Hassan Qadri. All rights reserved.
