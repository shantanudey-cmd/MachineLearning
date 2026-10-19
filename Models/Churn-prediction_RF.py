import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\BITS\GitHub\MachineLearningDataset")
filepath = folder / "churn.csv"

if filepath.is_file():
    churn = pd.read_csv(filepath)
    print(churn.shape)
    print(churn.head())
else:
    raise FileNotFoundError(f"File not found: {filepath}")
churn.describe()
# ---------- BLOCK 1: LOGISTIC REGRESSION ----------
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
import pickle

CSV = Path("C:/Users/BITS/GitHub/MachineLearningDataset/churn.csv")   # change if needed
TARGET = "Churn"

# 1. Load and clean
df = pd.read_csv(CSV)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
df = df.dropna(subset=[TARGET])
y = df[TARGET].astype(str).str.strip().str.lower().eq("yes").astype(int)  # Yes=1, No=0
X = df.drop(columns=[TARGET, "customerID"])                               # ID is not a feature

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# 2. Preprocessing: impute + scale numbers, impute + one-hot categories
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# 3. Model
model = Pipeline([("prep", prep),
                  ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

# 4. Split, train
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_tr, y_tr)

# 5. Evaluate
pred = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]
print("Accuracy :", round(accuracy_score(y_te, pred), 3))
print("Precision:", round(precision_score(y_te, pred), 3))
print("Recall   :", round(recall_score(y_te, pred), 3))
print("F1       :", round(f1_score(y_te, pred), 3))
print("ROC-AUC  :", round(roc_auc_score(y_te, proba), 3))
print("\nConfusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y_te, pred))
print("\n", classification_report(y_te, pred, target_names=["No churn", "Churn"]))

# 6. Top drivers of churn (coefficients)
names = model.named_steps["prep"].get_feature_names_out()
coef = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
print("Pushes towards STAYING :\n", coef.head(5).round(2))
print("Pushes towards CHURN   :\n", coef.tail(5).round(2))

# 7. Save
Path("Models").mkdir(exist_ok=True)
with open("Models/churn_logreg.pkl", "wb") as f:      # pickle: no extra install needed
    pickle.dump(model, f)
print("Saved Models/churn_logreg.pkl")