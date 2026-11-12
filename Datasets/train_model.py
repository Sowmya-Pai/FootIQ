
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load feature-engineered dataset
df = pd.read_csv("matches_with_features.csv")

# 2. Select input features
features = [
    "home_avg_points",
    "away_avg_points",
    "home_win_rate",
    "away_win_rate",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "home_previous_matches",
    "away_previous_matches"
]

X = df[features]
y = df["Target"]

# 3. Chronological train-test split (80:20)
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("--- DATA SPLIT ---")
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

# 4. Create a simple baseline model
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

# 5. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# 6. Train the model
model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\n--- MODEL PERFORMANCE ---")

print(f"Baseline Accuracy: {baseline_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

print("\n--- CLASSIFICATION REPORT ---")

print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=["Home Win", "Draw", "Away Win"],
    zero_division=0
))

print("\n--- CONFUSION MATRIX ---")

print(confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2]
))

# 9. Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- FEATURE IMPORTANCE ---")
print(importance.to_string(index=False))

# 10. Save trained model
joblib.dump(model, "footIQ_random_forest.pkl")

print("\nModel saved as footIQ_random_forest.pkl")
