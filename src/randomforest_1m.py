import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


df = pd.read_csv(
    "data/processed/final_dataset_1m.csv"
)

df["Date"] = pd.to_datetime(df["Date"])


split_date = "2023-01-01"

train_df = df[df["Date"] < split_date]
test_df = df[df["Date"] >= split_date]

print("Train Shape:")
print(train_df.shape)

print("\nTest Shape:")
print(test_df.shape)



exclude_cols = [
    "Date",
    "Stock",
    "Forward_1M_Return",
    "Target_1M"
]

features = [
    col for col in df.columns
    if col not in exclude_cols
]

print("\nFeatures Used:")
print(features)



X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["Target_1M"]
y_test = test_df["Target_1M"]



model = RandomForestClassifier(
    n_estimators=300,      #build 300 different decision trees in the forest
    max_depth=8,           #maximum number of split levels
    min_samples_leaf=20,
    random_state=42,       #Setting random_state=42, ensures that this randomness is identical every time you run the script 
    n_jobs=-1              #Setting this to -1 tells Python to utilize all available cores on your computer's processor simultaneously
)

model.fit(X_train, y_train)


preds = model.predict(X_test)

probs = model.predict_proba(X_test)[:, 1]


print("\nAccuracy:")
print(accuracy_score(y_test, preds))

print("\nROC-AUC:")
print(roc_auc_score(y_test, probs))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nTop Features:")
print(importance_df.head(20))



joblib.dump(    #dump function in joblib job is to take a Python object and convert it into a stream of bytes that can be written to a file
    model,
    "models/random_forest_1m.pkl"
)

print("\nModel Saved.")