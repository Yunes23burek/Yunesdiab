"""
============================================================================
Mini Project 15: Diabetes Prediction with Decision Trees & Logistic Regression
Author: Bourek Youness | M1 Microelectronics
University: Ferhat Abbas Setif-1 | Faculty of Technology
Course: Elements of Applied Artificial Intelligence
Date: May 2026
============================================================================

This script implements:
1. Decision Tree (Default)
2. Decision Tree (Optimized with GridSearchCV)
3. Logistic Regression

And compares their performance for diabetes prediction.
"""

# ============================================================
# Step 1: Import Required Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================
# Step 2: Load the Dataset
# ============================================================
print("=" * 70)
print("DIABETES PREDICTION - MINI PROJECT 15")
print("Bourek Youness | M1 Microelectronics | Ferhat Abbas Setif-1")
print("Faculty of Technology | Elements of Applied AI")
print("=" * 70)

# Load data (make sure diabetes.csv is in the same folder)
# Dataset source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
try:
    df = pd.read_csv('diabetes.csv')
    print("\n✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("\n❌ Error: diabetes.csv not found!")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    print("Or use the UCI repository:")
    print("https://archive.ics.uci.edu/dataset/34/diabetes")
    exit()

# ============================================================
# Step 3: Display Dataset Information
# ============================================================
print("\n📊 DATASET OVERVIEW")
print("-" * 50)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset Info:")
print(df.info())

print(f"\nStatistical Summary:")
print(df.describe())

print(f"\nMissing Values:")
print(df.isnull().sum())

print(f"\nClass Distribution:")
print(df['Outcome'].value_counts())
print(f"Non-Diabetic (0): {df['Outcome'].value_counts()[0]} ({df['Outcome'].value_counts()[0]/len(df)*100:.1f}%)")
print(f"Diabetic (1): {df['Outcome'].value_counts()[1]} ({df['Outcome'].value_counts()[1]/len(df)*100:.1f}%)")

# ============================================================
# Step 4: Data Visualization
# ============================================================
print("\n📈 GENERATING VISUALIZATIONS...")

# Figure 1: Distribution of Outcome
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
df['Outcome'].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('Diabetes Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Outcome (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
plt.pie(df['Outcome'].value_counts(), labels=['Non-Diabetic', 'Diabetic'], 
        autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90)
plt.title('Diabetes Proportion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation = df.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 3: Feature Distributions by Outcome
features = ['Glucose', 'BMI', 'Age', 'Insulin']
plt.figure(figsize=(14, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=feature, hue='Outcome', kde=True, 
                 palette=['#3498db', '#e74c3c'], alpha=0.6)
    plt.title(f'{feature} Distribution by Outcome', fontweight='bold')
plt.tight_layout()
plt.savefig('03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Step 5: Data Preprocessing
# ============================================================
print("\n🔧 DATA PREPROCESSING")
print("-" * 50)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Replace zeros with NaN for certain columns (biologically impossible to be 0)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    X[col] = X[col].replace(0, np.nan)
    X[col] = X[col].fillna(X[col].median())

print("✅ Zero values replaced with median for: Glucose, BloodPressure, SkinThickness, Insulin, BMI")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Step 6: Model 1 - Decision Tree (Default)
# ============================================================
print("\n🌳 MODEL 1: DECISION TREE (DEFAULT)")
print("-" * 50)

dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)
y_pred_dt1 = dt_default.predict(X_test)

print(f"Training Accuracy: {dt_default.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_dt1):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt1):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt1):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt1):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, dt_default.predict_proba(X_test)[:,1]):.4f}")

# ============================================================
# Step 7: Model 2 - Decision Tree (Optimized)
# ============================================================
print("\n🌳 MODEL 2: DECISION TREE (OPTIMIZED)")
print("-" * 50)

# Hyperparameter tuning with GridSearch
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

dt_optimized = grid_search.best_estimator_
y_pred_dt2 = dt_optimized.predict(X_test)

print(f"Training Accuracy: {dt_optimized.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_dt2):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt2):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt2):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt2):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, dt_optimized.predict_proba(X_test)[:,1]):.4f}")

# ============================================================
# Step 8: Model 3 - Logistic Regression
# ============================================================
print("\n📊 MODEL 3: LOGISTIC REGRESSION")
print("-" * 50)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print(f"Training Accuracy: {lr.score(X_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:,1]):.4f}")

# ============================================================
# Step 9: Visualize Decision Tree
# ============================================================
plt.figure(figsize=(20, 12))
plot_tree(dt_optimized, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'],
          filled=True, rounded=True, fontsize=10, max_depth=3)
plt.title('Optimized Decision Tree (Depth Limited to 3 for Visualization)', 
          fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('04_decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Step 10: Comparative Visualization
# ============================================================
print("\n📊 GENERATING COMPARATIVE VISUALIZATIONS...")

# Collect all metrics
models = ['DT (Default)', 'DT (Optimized)', 'Logistic Regression']

accuracy = [
    accuracy_score(y_test, y_pred_dt1),
    accuracy_score(y_test, y_pred_dt2),
    accuracy_score(y_test, y_pred_lr)
]

precision = [
    precision_score(y_test, y_pred_dt1),
    precision_score(y_test, y_pred_dt2),
    precision_score(y_test, y_pred_lr)
]

recall = [
    recall_score(y_test, y_pred_dt1),
    recall_score(y_test, y_pred_dt2),
    recall_score(y_test, y_pred_lr)
]

f1 = [
    f1_score(y_test, y_pred_dt1),
    f1_score(y_test, y_pred_dt2),
    f1_score(y_test, y_pred_lr)
]

auc = [
    roc_auc_score(y_test, dt_default.predict_proba(X_test)[:,1]),
    roc_auc_score(y_test, dt_optimized.predict_proba(X_test)[:,1]),
    roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:,1])
]

# Figure 4: Metrics Comparison Bar Chart
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=1.02)

metrics_data = [
    (accuracy, 'Accuracy', '#3498db'),
    (precision, 'Precision', '#2ecc71'),
    (recall, 'Recall', '#e74c3c'),
    (f1, 'F1-Score', '#9b59b6'),
    (auc, 'AUC-ROC', '#f39c12')
]

for idx, (values, title, color) in enumerate(metrics_data):
    ax = axes[idx // 3, idx % 3]
    bars = ax.bar(models, values, color=[color, color, color], alpha=0.8, edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)

# Feature Importance (6th subplot)
ax = axes[1, 2]
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_optimized.feature_importances_
}).sort_values('Importance', ascending=True)

ax.barh(importance['Feature'], importance['Importance'], color='#1abc9c')
ax.set_title('Feature Importance (Optimized DT)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig('05_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 5: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

predictions = [y_pred_dt1, y_pred_dt2, y_pred_lr]
for idx, (pred, name) in enumerate(zip(predictions, models)):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    axes[idx].set_title(name, fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('06_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 6: ROC Curves
plt.figure(figsize=(10, 8))

# DT Default
fpr1, tpr1, _ = roc_curve(y_test, dt_default.predict_proba(X_test)[:,1])
plt.plot(fpr1, tpr1, label=f'DT (Default) - AUC = {auc[0]:.3f}', linewidth=2)

# DT Optimized
fpr2, tpr2, _ = roc_curve(y_test, dt_optimized.predict_proba(X_test)[:,1])
plt.plot(fpr2, tpr2, label=f'DT (Optimized) - AUC = {auc[1]:.3f}', linewidth=2)

# Logistic Regression
fpr3, tpr3, _ = roc_curve(y_test, lr.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr3, tpr3, label=f'Logistic Regression - AUC = {auc[2]:.3f}', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('07_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Step 11: Final Summary Table
# ============================================================
print("\n" + "=" * 70)
print("📋 FINAL RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame({
    'Model': models,
    'Accuracy': [f'{a:.4f}' for a in accuracy],
    'Precision': [f'{p:.4f}' for p in precision],
    'Recall': [f'{r:.4f}' for r in recall],
    'F1-Score': [f'{f:.4f}' for f in f1],
    'AUC-ROC': [f'{a:.4f}' for a in auc]
})

print(results_df.to_string(index=False))

# Best model
best_idx = np.argmax(f1)
print(f"\n🏆 BEST MODEL (based on F1-Score): {models[best_idx]}")
print(f"   F1-Score: {f1[best_idx]:.4f}")

# ============================================================
# Step 12: Conclusion
# ============================================================
print("\n" + "=" * 70)
print("📝 CONCLUSION")
print("=" * 70)

conclusion = """
1. DATASET: The Pima Indians Diabetes dataset contains 768 samples with 8 features.
   The dataset is slightly imbalanced with ~65% non-diabetic and ~35% diabetic cases.

2. PREPROCESSING: Zero values in biological features (Glucose, BloodPressure, etc.)
   were replaced with median values to handle missing data appropriately.

3. MODELS COMPARED:
   • Decision Tree (Default): Prone to overfitting with 100% training accuracy
   • Decision Tree (Optimized): GridSearch found optimal parameters, reducing overfitting
   • Logistic Regression: Stable baseline model with good generalization

4. KEY FINDINGS:
   • Glucose level is the most important feature for diabetes prediction
   • BMI and Age are also significant predictors
   • The optimized Decision Tree provides a good balance between interpretability and performance
   • Logistic Regression offers stable performance with probabilistic outputs

5. RECOMMENDATION: For medical diagnosis, the Optimized Decision Tree or Logistic
   Regression are preferred due to their interpretability and reliable performance.
"""
print(conclusion)

print("\n✅ Project completed successfully!")
print("👨‍💻 Author: Bourek Youness | M1 Microelectronics | Ferhat Abbas Setif-1")
print("📚 Course: Elements of Applied Artificial Intelligence")
print("🏫 Faculty of Technology")
