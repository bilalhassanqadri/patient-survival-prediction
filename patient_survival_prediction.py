# =============================================================================
# Patient Survival and Risk Level Prediction
# MSc Dissertation — University of Hertfordshire (Distinction)
# Author: Bilal Hassan Qadri
#
# This script builds a complete machine learning pipeline to predict whether
# a patient will survive or not based on their clinical ICU data. I went
# through 14 steps in my dissertation — this code covers all of them.
#
# The main idea: instead of relying on one classifier, I trained seven
# different models and combined them into a single ensemble that votes on
# the final prediction. This approach consistently outperformed any
# individual model on its own.
#
# Dataset: ICU patient records with ~106 features after preprocessing
# Target : binary label — 0 (did not survive) | 1 (survived)
# =============================================================================


# -----------------------------------------------------------------------------
# Imports
# I have grouped these by purpose so it is easy to see what each block does
# -----------------------------------------------------------------------------

# Core data handling
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')   # suppress sklearn version warnings

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Preprocessing tools
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.decomposition    import PCA

# Model selection and evaluation
from sklearn.model_selection  import (train_test_split,
                                      StratifiedKFold,
                                      cross_validate)
from sklearn.metrics          import (confusion_matrix,
                                      ConfusionMatrixDisplay,
                                      accuracy_score,
                                      precision_score,
                                      recall_score,
                                      f1_score,
                                      classification_report)

# The seven classifiers I compared in my dissertation
from sklearn.linear_model     import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.naive_bayes       import GaussianNB
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.ensemble          import (GradientBoostingClassifier,
                                       RandomForestClassifier,
                                       VotingClassifier)
from sklearn.svm               import SVC

# Handling class imbalance — a major challenge in clinical datasets
from imblearn.combine          import SMOTETomek


# =============================================================================
# STEP 1 — Load the dataset and remove columns we do not need
# =============================================================================
# The raw dataset contains several identifier columns (patient ID, hospital ID
# etc.) that carry no predictive information. There are also duplicate columns
# for noninvasive measurements which I identified and removed to avoid
# redundancy. The 'Unnamed: 83' column is an artefact from the CSV export.

print("\n" + "=" * 65)
print("STEP 1: Loading dataset")
print("=" * 65)

df = pd.read_csv('dataset.csv')
print(f"  Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Drop identifier columns — these are row labels, not features
id_cols = ['encounter_id', 'patient_id', 'hospital_id', 'Unnamed: 83', 'icu_id']
df.drop([c for c in id_cols if c in df.columns], axis=1, inplace=True)

# Drop noninvasive duplicate columns
# These columns record the same vital signs measured a different way.
# Keeping both would introduce multicollinearity without adding information.
noninvasive_cols = [c for c in df.columns if 'noninvasive' in c.split('_')]
df.drop(noninvasive_cols, axis=1, inplace=True)

print(f"  After removing identifiers and duplicates: {df.shape[0]} rows x {df.shape[1]} columns")


# =============================================================================
# STEP 2 — Separate features from the target variable
# =============================================================================
# The last column is the binary survival label.
# Everything else is a clinical feature (vitals, labs, demographics, etc.)

print("\n" + "=" * 65)
print("STEP 2: Separating features and target")
print("=" * 65)

y = df.iloc[:, -1].copy()     # target: 0 = did not survive, 1 = survived
x = df.iloc[:, :-1].copy()   # features: all clinical measurements

print(f"  Feature matrix shape : {x.shape}")
print(f"  Target distribution  :\n{y.value_counts().to_string()}")
print(f"\n  Class imbalance ratio: {round(y.value_counts()[0] / y.value_counts()[1], 2)}:1")
print("  (This imbalance is why we need SMOTETomek in Step 6)")


# =============================================================================
# STEP 3 — Handle categorical columns: impute then encode
# =============================================================================
# Clinical datasets mix numeric measurements with categorical variables like
# gender, ethnicity, ICU type, admission source, etc. Machine learning models
# need everything to be numeric, so I:
#   (a) fill missing categorical values with the most common value
#   (b) one-hot encode each category into separate binary columns

print("\n" + "=" * 65)
print("STEP 3: Categorical imputation and one-hot encoding")
print("=" * 65)

cat_cols = x.select_dtypes('object').columns.tolist()
print(f"  Found {len(cat_cols)} categorical columns")

# (a) Impute missing categorical values
cat_missing = [c for c in cat_cols if x[c].isna().any()]
if cat_missing:
    print(f"  Imputing {len(cat_missing)} categorical columns with most frequent value")
    impute_cat = SimpleImputer(strategy='most_frequent')
    x[cat_missing] = impute_cat.fit_transform(x[cat_missing])

# (b) One-hot encode — converts each category into a binary 0/1 column
# handle_unknown='ignore' means if we see an unseen category at test time,
# it just gets encoded as all zeros rather than crashing
encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
x_encoded   = encoder.fit_transform(x[cat_cols])
df_encoded  = pd.DataFrame(
    x_encoded,
    columns=encoder.get_feature_names_out(cat_cols),
    index=x.index
)

# Merge encoded columns in and drop the original string columns
x = pd.concat([x.drop(columns=cat_cols), df_encoded], axis=1)
print(f"  Shape after encoding: {x.shape[0]} rows x {x.shape[1]} columns")


# =============================================================================
# STEP 4 — Handle numeric missing values
# =============================================================================
# Not all missing values are equal. Columns with very few missing values
# (under 3%) are safe to drop those rows without losing much data. Columns
# with more missingness need imputation — I use the median because it is
# robust to outliers, which are common in ICU lab measurements.

print("\n" + "=" * 65)
print("STEP 4: Numeric imputation")
print("=" * 65)

miss_pct  = x.isna().mean() * 100
cols_lt3  = miss_pct[miss_pct < 3].index.tolist()
cols_gt3  = miss_pct[miss_pct >= 3].index.tolist()

print(f"  Columns with < 3% missing (drop rows): {len(cols_lt3)}")
print(f"  Columns with >= 3% missing (impute)  : {len(cols_gt3)}")

# Drop rows only for the low-missingness columns
combined = pd.concat([x, y], axis=1)
combined.dropna(subset=cols_lt3, inplace=True)
x = combined.iloc[:, :-1].copy()
y = combined.iloc[:,  -1].copy()

# Median imputation for everything else
# Median is better than mean here because ICU measurements (creatinine,
# bilirubin, etc.) often have extreme outliers that skew the mean
impute_num = SimpleImputer(strategy='median')
x = pd.DataFrame(
    impute_num.fit_transform(x),
    columns=x.columns,
    index=x.index
)
print(f"  Shape after numeric imputation: {x.shape}")


# =============================================================================
# STEP 5 — Train / test split  (this MUST happen before SMOTETomek)
# =============================================================================
# One of the most important decisions in this pipeline: the split comes BEFORE
# resampling. If you apply SMOTETomek on the full dataset first and then split,
# synthetic samples from the minority class will leak into the test set. That
# means your accuracy score measures performance on partially synthetic data,
# not real patients — which overstates how well the model generalises.
#
# Correct order: split → resample training data only → scale → PCA → train

print("\n" + "=" * 65)
print("STEP 5: Stratified train / test split (70% train, 30% test)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size    = 0.3,
    random_state = 42,
    stratify     = y    # preserves class ratio in both splits
)

print(f"  Training set : {X_train.shape[0]} samples")
print(f"  Test set     : {X_test.shape[0]} samples")
print(f"  Train class balance:\n{pd.Series(y_train).value_counts().to_string()}")


# =============================================================================
# STEP 6 — SMOTETomek resampling (training data only)
# =============================================================================
# ICU survival datasets are inherently imbalanced — there are usually far more
# survivors than non-survivors. Without correction, classifiers learn to just
# predict the majority class and still get decent accuracy, while completely
# failing on the minority class that matters most clinically.
#
# SMOTETomek is a two-part technique:
#   SMOTE     — creates synthetic minority class samples by interpolating
#               between existing minority samples in feature space
#   Tomek Links — removes majority class samples that are too close to the
#               decision boundary (cleaning borderline cases)
#
# Together they both oversample the minority AND clean the majority class,
# giving the model a much cleaner decision boundary to learn from.

print("\n" + "=" * 65)
print("STEP 6: SMOTETomek resampling — training data only")
print("=" * 65)

smot = SMOTETomek(random_state=42)
X_train, y_train = smot.fit_resample(X_train, y_train)

print(f"  After resampling — training samples: {X_train.shape[0]}")
print(f"  Resampled class balance:\n{pd.Series(y_train).value_counts().to_string()}")


# =============================================================================
# STEP 7 — Feature scaling with StandardScaler
# =============================================================================
# Many classifiers (especially SVM and KNN) are sensitive to feature scale.
# A blood pressure reading in mmHg (range 60-200) would otherwise dominate
# a binary encoded feature (range 0-1) simply due to magnitude, not importance.
#
# StandardScaler transforms each feature to have mean=0 and std=1.
# Critical rule: fit the scaler ONLY on training data, then transform both.
# Fitting on the full dataset would let test-set statistics influence the
# scaler — another form of data leakage.

print("\n" + "=" * 65)
print("STEP 7: StandardScaler (fit on training data only)")
print("=" * 65)

scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_test_scaled   = scaler.transform(X_test)    # transform only — never fit on test

print("  Scaling complete")
print(f"  Training feature means (first 5): "
      f"{X_train_scaled[:, :5].mean(axis=0).round(4)}")
print("  (Should all be approximately 0.0 after scaling)")


# =============================================================================
# STEP 8 — Dimensionality reduction with PCA
# =============================================================================
# After encoding, we have over 100 features. Many of these are correlated
# (e.g., multiple blood pressure readings, multiple glucose measurements).
# PCA reduces this to a smaller set of uncorrelated components that capture
# most of the variance in the data, which:
#   - speeds up model training
#   - reduces risk of overfitting on high-dimensional sparse data
#   - removes multicollinearity that can destabilise some classifiers
#
# Rather than hardcoding a number of components, I fit a full PCA and use
# a scree plot to choose the number of components that explain 90% of
# the total variance. This makes the choice data-driven and reproducible.

print("\n" + "=" * 65)
print("STEP 8: PCA — selecting components via explained variance")
print("=" * 65)

# Quick 2D and 3D visualisations to check if classes are separable in PCA space
pca_2d    = PCA(n_components=2)
X_2d      = pca_2d.fit_transform(X_train_scaled)
fig_2d    = px.scatter(
    x=X_2d[:, 0], y=X_2d[:, 1],
    color=y_train.astype(str),
    title='PCA 2-D Projection of Training Data',
    labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
    color_discrete_sequence=px.colors.qualitative.G10
)
fig_2d.show()

pca_3d    = PCA(n_components=3)
X_3d      = pca_3d.fit_transform(X_train_scaled)
fig_3d    = px.scatter_3d(
    x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
    color=y_train.astype(str),
    title='PCA 3-D Projection of Training Data'
)
fig_3d.update_layout(margin=dict(l=20, r=20, t=20, b=20))
fig_3d.show()

# Fit full PCA to see the complete variance distribution
pca_full  = PCA()
pca_full.fit(X_train_scaled)
cum_var   = np.cumsum(pca_full.explained_variance_ratio_)

# Plot scree plot and cumulative variance curve side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(pca_full.explained_variance_ratio_[:80],
             marker='o', markersize=3, color='steelblue')
axes[0].axhline(0.01, color='red', linestyle='--', alpha=0.7, label='1% threshold')
axes[0].set_title('Scree Plot — Individual Explained Variance per Component')
axes[0].set_xlabel('Principal Component Index')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].legend()

axes[1].plot(cum_var[:120], marker='o', markersize=3, color='darkorange')
axes[1].axhline(0.90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
axes[1].set_title('Cumulative Explained Variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].legend()

plt.suptitle('PCA Component Selection Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Select the minimum number of components that explain >= 90% variance
n_components = int(np.argmax(cum_var >= 0.90)) + 1
print(f"  Components explaining 90% variance: {n_components}")

# Apply final PCA transformation
pca          = PCA(n_components=n_components)
X_train_pca  = pca.fit_transform(X_train_scaled)
X_test_pca   = pca.transform(X_test_scaled)

print(f"  Reduced training shape : {X_train_pca.shape}")
print(f"  Reduced test shape     : {X_test_pca.shape}")


# =============================================================================
# STEP 9 — Train and evaluate seven individual classifiers
# =============================================================================
# I selected these seven classifiers because they represent a diverse range
# of learning strategies:
#
#   LR   — linear decision boundary, good baseline
#   DT   — non-linear, interpretable, prone to overfitting alone
#   GNB  — probabilistic, fast, works well with independent features
#   KNN  — instance-based, no explicit training, sensitive to scale
#   GB   — boosting: sequential weak learners, strong in practice
#   RF   — bagging: parallel decision trees, reduces variance
#   SVM  — finds maximum-margin hyperplane, strong on high-dim data
#
# Each model is evaluated with:
#   (a) 10-fold stratified cross-validation on training data
#       → gives mean ± std accuracy, which is more reliable than one split
#   (b) Final evaluation on the held-out test set
#       → gives the honest generalisation performance

print("\n" + "=" * 65)
print("STEP 9: Individual classifiers — training and evaluation")
print("=" * 65)

models = {
    'Logistic Regression'    : LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'          : DecisionTreeClassifier(random_state=42),
    'Gaussian Naive Bayes'   : GaussianNB(),
    'K-Nearest Neighbor'     : KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting'      : GradientBoostingClassifier(
                                   n_estimators  = 100,
                                   learning_rate = 1.0,
                                   max_depth     = 1,
                                   random_state  = 42),
    'Random Forest'          : RandomForestClassifier(max_depth=2, random_state=42),
    'Support Vector Machine' : SVC(probability=True, random_state=42),
}

# 10-fold stratified CV — stratified means each fold keeps the class ratio
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Storage for the final comparison table
results = {
    'Model'       : [],
    'CV Acc Mean' : [],
    'CV Acc Std'  : [],
    'Test Acc'    : [],
    'Precision'   : [],
    'Recall'      : [],
    'F1'          : [],
}

# Store fitted models — we need them later for the ensemble
trained_models = {}

for name, model in models.items():
    print(f"\n  ── {name}")

    # (a) Cross-validation
    cv_out = cross_validate(
        model, X_train_pca, y_train,
        cv      = cv,
        scoring = ['accuracy', 'precision', 'recall', 'f1'],
        n_jobs  = -1
    )
    cv_mean = cv_out['test_accuracy'].mean()
    cv_std  = cv_out['test_accuracy'].std()
    print(f"     10-Fold CV  : {cv_mean:.4f} ± {cv_std:.4f}")

    # (b) Fit on full training set, predict on held-out test set
    model.fit(X_train_pca, y_train)
    trained_models[name] = model
    y_pred = model.predict(X_test_pca)

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score   (y_test, y_pred, zero_division=0)
    f1   = f1_score       (y_test, y_pred, zero_division=0)

    print(f"     Test Acc    : {acc:.4f}")
    print(f"     Precision   : {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=['Did Not Survive', 'Survived'],
                                zero_division=0))

    # Confusion matrix — shows where the model makes mistakes
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Did Not Survive', 'Survived'])
    disp.plot(cmap='Blues')
    plt.title(f'{name} — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'cm_{name.lower().replace(" ", "_")}.png', dpi=120)
    plt.show()

    # Save to results
    results['Model']      .append(name)
    results['CV Acc Mean'].append(round(cv_mean, 4))
    results['CV Acc Std'] .append(round(cv_std,  4))
    results['Test Acc']   .append(round(acc,      4))
    results['Precision']  .append(round(prec,     4))
    results['Recall']     .append(round(rec,       4))
    results['F1']         .append(round(f1,        4))


# =============================================================================
# STEP 10 — Proposed EV-Patient Survival Ensemble (soft VotingClassifier)
# =============================================================================
# A VotingClassifier combines multiple trained models. In 'soft' voting mode,
# each model outputs a probability for each class and those probabilities are
# averaged. The class with the highest average probability wins.
#
# Why soft over hard voting?
#   Hard voting: each model casts a binary vote (0 or 1), majority wins
#   Soft voting : each model contributes its confidence level
#   → A model that is 99% sure outweighs one that is 51% sure
#   → This leads to better calibrated and generally more accurate predictions
#
# The intuition behind ensemble voting:
#   SVM might struggle where the decision boundary is non-linear near the edge
#   Random Forest compensates with its diverse tree structure
#   Gradient Boosting adds sequential error correction
#   Together they cover each other's blind spots

print("\n" + "=" * 65)
print("STEP 10: Proposed EV-Patient Survival — Ensemble Voting Classifier")
print("=" * 65)

ensemble = VotingClassifier(
    estimators = [(name.lower().replace(' ', '_'), mdl)
                  for name, mdl in trained_models.items()],
    voting     = 'soft'
)

# Cross-validate the ensemble just like the individual models
cv_ens     = cross_validate(
    ensemble, X_train_pca, y_train,
    cv      = cv,
    scoring = ['accuracy', 'precision', 'recall', 'f1'],
    n_jobs  = -1
)
ens_cv_mean = cv_ens['test_accuracy'].mean()
ens_cv_std  = cv_ens['test_accuracy'].std()
print(f"  10-Fold CV  : {ens_cv_mean:.4f} ± {ens_cv_std:.4f}")

# Fit on full training set and evaluate on test set
ensemble.fit(X_train_pca, y_train)
y_pred_ens = ensemble.predict(X_test_pca)

ens_acc  = accuracy_score (y_test, y_pred_ens)
ens_prec = precision_score(y_test, y_pred_ens, zero_division=0)
ens_rec  = recall_score   (y_test, y_pred_ens, zero_division=0)
ens_f1   = f1_score       (y_test, y_pred_ens, zero_division=0)

print(f"  Test Acc    : {ens_acc:.4f}")
print(f"  Precision   : {ens_prec:.4f}  |  Recall: {ens_rec:.4f}  |  F1: {ens_f1:.4f}")
print(classification_report(y_test, y_pred_ens,
                             target_names=['Did Not Survive', 'Survived'],
                             zero_division=0))

# Ensemble confusion matrix
cm_ens   = confusion_matrix(y_test, y_pred_ens)
disp_ens = ConfusionMatrixDisplay(confusion_matrix=cm_ens,
                                   display_labels=['Did Not Survive', 'Survived'])
disp_ens.plot(cmap='Greens')
plt.title('EV-Patient Survival Ensemble — Confusion Matrix')
plt.tight_layout()
plt.savefig('cm_ensemble.png', dpi=120)
plt.show()

# Add ensemble to results table
results['Model']      .append('EV-Patient Survival (Ensemble)')
results['CV Acc Mean'].append(round(ens_cv_mean, 4))
results['CV Acc Std'] .append(round(ens_cv_std,  4))
results['Test Acc']   .append(round(ens_acc,      4))
results['Precision']  .append(round(ens_prec,     4))
results['Recall']     .append(round(ens_rec,       4))
results['F1']         .append(round(ens_f1,        4))


# =============================================================================
# STEP 11 — Summary table and comparison charts
# =============================================================================
# Everything above produces numbers. This section turns those numbers into
# the charts and tables that go into the paper / dissertation appendix.

print("\n" + "=" * 65)
print("STEP 11: Summary and visualisations")
print("=" * 65)

df_results = pd.DataFrame(results)

print("\n  FINAL RESULTS — ALL MODELS")
print("  " + "-" * 63)
print(df_results.to_string(index=False))
print("  " + "-" * 63)

# Save results so we can paste into the paper
df_results.to_csv('model_results_summary.csv', index=False)
print("\n  Saved: model_results_summary.csv")

# Colour the ensemble bar differently so it stands out
bar_colours = ['#2563eb' if m != 'EV-Patient Survival (Ensemble)'
               else '#16a34a' for m in df_results['Model']]

# Four-panel metric comparison
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
metrics = [
    ('Test Acc',  'Test Set Accuracy'),
    ('Precision', 'Precision'),
    ('Recall',    'Recall'),
    ('F1',        'F1-Score'),
]

for ax, (col, title) in zip(axes.flat, metrics):
    bars = ax.barh(df_results['Model'], df_results[col],
                   color=bar_colours, edgecolor='white', linewidth=0.4)
    ax.set_xlim(0, 1.08)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Score')
    ax.axvline(0.9, color='red', linestyle='--', alpha=0.4, label='0.90 reference')
    for bar, val in zip(bars, df_results[col]):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8.5)
    ax.legend(fontsize=8)

plt.suptitle('Model Performance Comparison — Patient Survival Prediction',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: model_comparison.png")

# Cross-validation accuracy with error bars
fig, ax = plt.subplots(figsize=(13, 6))
x_pos = np.arange(len(df_results))
ax.bar(x_pos, df_results['CV Acc Mean'],
       yerr      = df_results['CV Acc Std'],
       capsize   = 5,
       color     = bar_colours,
       edgecolor = 'white',
       linewidth = 0.4)
ax.set_xticks(x_pos)
ax.set_xticklabels(df_results['Model'], rotation=30, ha='right', fontsize=10)
ax.set_ylabel('10-Fold CV Accuracy (Mean ± Std)')
ax.set_ylim(0, 1.08)
ax.set_title('Cross-Validation Accuracy Comparison — All Models',
             fontsize=13, fontweight='bold')
ax.axhline(0.9, color='red', linestyle='--', alpha=0.4, label='0.90 reference')
ax.legend()
plt.tight_layout()
plt.savefig('cv_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: cv_accuracy_comparison.png")

print("\n" + "=" * 65)
print("Pipeline finished. Files generated:")
print("  model_results_summary.csv")
print("  model_comparison.png")
print("  cv_accuracy_comparison.png")
print("  pca_analysis.png")
print("  cm_<model_name>.png  (one per model + ensemble)")
print("=" * 65)
