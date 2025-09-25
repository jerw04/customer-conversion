import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
uci = pd.read_csv("data/e-shop clothing 2008.csv", sep=";")

# Rename columns
uci = uci.rename(columns={
    "session ID": "session_id",
    "page 1 (main category)": "page1_main_category",
    "page 2 (clothing model)": "page2_clothing_model",
    "model photography": "model_photography",
    "price 2": "price_2"
})

# Add target
uci["converted"] = (uci["page"] == 4).astype(int)

session_df = uci.groupby("session_id").agg({
    "month": "first",
    "day": "first",
    "country": "first",
    "page1_main_category": lambda x: x.mode()[0],
    "page2_clothing_model": lambda x: x.nunique(),
    "colour": lambda x: x.nunique(),
    "location": lambda x: x.nunique(),
    "model_photography": lambda x: x.mode()[0],
    "price": ["mean","max"],
    "price_2": "mean",
    "page": "max",
    "converted": "max"
}).reset_index()

# 🔹 Flatten MultiIndex column names
session_df.columns = [
    "_".join([str(c) for c in col if c]) for col in session_df.columns
]

# 🔹 Clean them up
session_df.columns = (
    pd.Index(session_df.columns)
    .str.replace("<lambda>", "agg", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)


# Features / target
X = session_df.drop(columns=["session_id", "converted_max"])
y = session_df["converted_max"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save locally (compatible with your Python version)
joblib.dump(rf, "rf_model.pkl")
print("✅ Random Forest model saved as rf_model.pkl")
