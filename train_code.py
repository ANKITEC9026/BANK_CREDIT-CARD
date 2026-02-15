import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ================= LOAD DATA =================
print("Loading dataset...")
df = pd.read_csv("bank-additional-full.csv", sep=None, engine="python")

# Clean column names
df.columns = df.columns.str.strip()

print("Columns detected:")
print(df.columns.tolist())

# ================= FIX TARGET COLUMN =================
target_col = "y"   # Correct target for bank dataset
print("Using target column:", target_col)

# Convert target to 0/1
df[target_col] = df[target_col].astype(str).str.lower().map({"yes": 1, "no": 0})

# ================= HANDLE UNKNOWN / MISSING =================
df.replace("unknown", np.nan, inplace=True)

# Fill numeric columns
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ================= ENCODING =================
df = pd.get_dummies(df)

# ================= SAVE STRUCTURE =================
os.makedirs("model", exist_ok=True)
joblib.dump(df.columns, "model/columns.pkl")

# ================= SPLIT FEATURES =================
X = df.drop(target_col, axis=1)
y = df[target_col]

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "model/scaler.pkl")

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ================= MODELS =================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1200),
    "Decision Tree": DecisionTreeClassifier(max_depth=6),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=120),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

# ================= METRIC FUNCTION =================
def evaluate(model):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "MCC": matthews_corrcoef(y_test, preds)
    }

# ================= TRAIN + SAVE =================
results = []

print("\nTraining models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, f"model/{name.replace(' ', '_')}.pkl")

    # Evaluate
    scores = evaluate(model)
    scores["Model"] = name
    results.append(scores)

    print(f"{name} trained & saved")

# ================= CREATE COMPARISON TABLE =================
results_df = pd.DataFrame(results)
results_df = results_df[["Model","Accuracy","AUC","Precision","Recall","F1","MCC"]]

results_df.to_csv("model/model_metrics.csv", index=False)

print("\n=== FINAL MODEL PERFORMANCE ===")
print(results_df.sort_values(by="F1", ascending=False))

print("\nAll models trained successfully!")
