# Patient Survival & Risk Level Prediction using Hyperparameter-Optimised Ensemble Learning

> **MSc Dissertation Project — University of Hertfordshire, UK (Distinction)**
> Bilal Hassan Qadri | MSc Artificial Intelligence and Robotics

---

## Overview

This project presents an end-to-end machine learning pipeline for predicting **patient survival and risk level** in clinical settings. The pipeline combines rigorous data preprocessing, class imbalance handling, dimensionality reduction, hyperparameter optimisation, and an ensemble voting classifier to assist healthcare professionals in making informed decisions about patient care.

A cross-platform mobile/web deployment interface was also built to allow clinicians to input patient data and receive a real-time survival prediction (0 = did not survive, 1 = survived).

---


## Key Results

| Model | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (LR)** | 93.78% | 0.930 | 0.940 | 0.9379 |
| **Decision Tree (DT)** | 90.12% | 0.890 | 0.920 | 0.9027 |
| **Gaussian Naive Bayes (GNB)** | 81.63% | 0.778 | 0.883 | 0.8275 |
| **K-Nearest Neighbor (KNN)** | 81.63% | 0.778 | 0.883 | 0.8275 |
| **Gradient Boosting (GB)** | 91.99% | 0.915 | 0.924 | 0.9200 |
| **Random Forest (RF)** | 81.88% | 0.809 | 0.832 | 0.8209 |
| **Support Vector Machine (SVM)** | 99.31% | 0.989 | 0.997 | 0.9931 |
| **EV-Patient Survival (Ensemble)** | **99.31%** | **0.989** | **0.997** | **0.9931** |

---


## Pipeline Architecture

```
Raw Clinical Dataset
        │
        ▼
Step 1  ─ Load data, drop identifier & duplicate columns
        │
        ▼
Step 2  ─ Separate features (X) and target (y)
        │
        ▼
Step 3  ─ Categorical imputation (most-frequent) + One-Hot Encoding
        │
        ▼
Step 4  ─ Numeric imputation (median strategy, split by 3% missingness threshold)
        │
        ▼
Step 5  ─ Stratified Train / Test Split (70 / 30)  ← split BEFORE resampling
        │
        ▼
Step 6  ─ SMOTETomek resampling — applied to TRAINING DATA ONLY
        │
        ▼
Step 7  ─ StandardScaler (fit on train, transform test)
        │
        ▼
Step 8  ─ PCA with scree plot — n_components selected at 90% explained variance
        │
        ▼
Step 9  ─ 7 individual classifiers trained + 10-fold stratified cross-validation
        │
        ▼
Step 10 ─ Proposed EV-Patient Survival ensemble (soft VotingClassifier)
        │
        ▼
Step 11 ─ Comparative visualisations + results summary CSV
```

---

## Methodology Highlights

### Class Imbalance — SMOTETomek
Clinical datasets are inherently imbalanced (more survivors than non-survivors). This pipeline uses **SMOTETomek**, a hybrid technique combining SMOTE oversampling of the minority class with Tomek Links removal of borderline majority samples. Critically, resampling is applied **only to training data** to prevent data leakage into the test set.

### Dimensionality Reduction — PCA
With 106 original features after encoding, Principal Component Analysis (PCA) is applied after scaling. The number of components is selected automatically via a **scree plot and cumulative explained variance threshold (90%)**, rather than being hardcoded. This ensures the choice is data-driven and reproducible.

### Hyperparameter Optimisation
A combined **Grid Search + Sequential Search** strategy was used to optimise hyperparameters across all 7 classifiers. This balances exhaustive search coverage with computational efficiency.

### Ensemble Voting Classifier
The proposed **EV-Patient Survival** model combines predictions from all 7 trained classifiers using **soft voting** (probability-weighted), which outperforms hard voting by leveraging model confidence scores.

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

This project uses a publicly available clinical ICU dataset. Due to data use agreements, the raw `dataset.csv` is not included in this repository.

To replicate this project, you may use the **MIMIC-III** dataset (requires free registration):
- Apply for access at: https://physionet.org/content/mimiciii/

Or use the **WiDS Datathon 2020** dataset (patient survival, publicly available):
- Download at: https://www.kaggle.com/competitions/widsdatathon2020/data

Place your dataset file as `dataset.csv` in the project root before running.

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

A cross-platform mobile application was developed alongside this pipeline to provide clinicians with a real-time interface. The app accepts patient feature inputs and returns a binary prediction:

- **0** — Patient at high risk / predicted not to survive
- **1** — Patient predicted to survive

The application was built using **Flutter (Dart)** with a Python backend serving the trained ensemble model.

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
