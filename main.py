# -------------------- SETUP --------------------
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pickle

from google.colab import files
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# -------------------- FOLDER CREATION --------------------
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("feature/eda", exist_ok=True)
os.makedirs("feature/ml", exist_ok=True)
os.makedirs("model", exist_ok=True)

# -------------------- UPLOAD CSV FILE --------------------
print("Please upload your dataset CSV file")
uploaded = files.upload()

# Move uploaded file to correct folder
for filename in uploaded.keys():
    shutil.move(filename, f"data/raw/{filename}")

# Load dataset
file_path = f"data/raw/{filename}"
df = pd.read_csv(file_path)

# -------------------- EDA --------------------
print("\nBasic Info:")
print(df.info())

print("\nDescription:")
print(df.describe())

print(df.head())

# Churn Count
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("feature/eda/churn_distribution.png")
plt.show()

# Gender vs Churn
if 'Gender' in df.columns:
    sns.countplot(x="Gender", hue="Churn", data=df)
    plt.title("Gender-wise Churn Distribution")
    plt.savefig("feature/eda/gender_churn.png")
    plt.show()

# Age distribution
if 'Age' in df.columns:
    sns.histplot(df['Age'], kde=True)
    plt.title("Age Distribution")
    plt.savefig("feature/eda/age_distribution.png")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("feature/eda/correlation_matrix.png")
plt.show()

# -------------------- FEATURE ENGINEERING --------------------
# Drop CustomerID if exists
if "CustomerID" in df.columns:
    df.drop("CustomerID", axis=1, inplace=True)

# Replace binary text values with 1/0
df.replace({"Yes": 1, "No": 0}, inplace=True)

# One-Hot Encoding for all categorical columns except target 'Churn'
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'Churn']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save cleaned data
df.to_csv("data/processed/churn_cleaned.csv", index=False)

# -------------------- NORMALIZATION --------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Churn' in numeric_cols:
    numeric_cols.remove('Churn')

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------------------- TRAIN-TEST SPLIT --------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# -------------------- BASELINE RANDOM FOREST --------------------
baseline_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
baseline_model.fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))
print(f"\nBaseline Model Accuracy: {baseline_accuracy:.4f}")

# -------------------- HYPERPARAMETER TUNING --------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# -------------------- FEATURE SELECTION --------------------
selector = SelectFromModel(best_model, threshold="median")
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# -------------------- FINAL MODEL TRAINING --------------------
best_model.fit(X_train_sel, y_train)
y_pred = best_model.predict(X_test_sel)

# -------------------- EVALUATION --------------------
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print(f"\nFinal Model Accuracy: {accuracy:.4f}")

# Save metrics
with open("model/model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)

# -------------------- FEATURE IMPORTANCE --------------------
selected_features = X.columns[selector.get_support()]
importances = pd.Series(best_model.feature_importances_, index=selected_features)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(8, 6))
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("feature/ml/feature_importance.png")
plt.show()

# -------------------- SAVE MODEL --------------------
with open("model/churn_model_improved.pkl", "wb") as f:
    pickle.dump(best_model, f)

# -------------------- SAMPLE PREDICTIONS --------------------
print("All steps completed. Model and plots saved.")

sample_indices = np.random.choice(X_test_sel.shape[0], size=5, replace=False)
sample_data = X_test_sel[sample_indices]
sample_true_labels = y_test.iloc[sample_indices]
sample_preds = best_model.predict(sample_data)

print("\nSample Predictions:")
for i in range(len(sample_preds)):
    print(f"Sample {i+1} | Actual: {sample_true_labels.iloc[i]} | Predicted: {sample_preds[i]}")

print(df.head())
