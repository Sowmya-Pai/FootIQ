
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load advanced dataset
df = pd.read_csv("matches_with_advanced_features.csv")

# 2. Select all 17 features
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
    "away_previous_matches",
    "home_home_avg_points",
    "away_away_avg_points",
    "home_home_avg_goal_diff",
    "away_away_avg_goal_diff",
    "h2h_home_avg_points",
    "h2h_home_avg_goal_diff",
    "elo_difference"
]

X = df[features]
y = df["Target"]

# 3. Chronological 80:20 split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("--- DATA SPLIT ---")
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

# 4. Baseline model
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

# 5. Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# 6. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\n--- MODEL PERFORMANCE ---")
print(f"Baseline Accuracy: {baseline_accuracy * 100:.2f}%")
print(f"Version 2 Accuracy: {accuracy * 100:.2f}%")

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Home Win", "Draw", "Away Win"],
        zero_division=0
    )
)

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

# 7. Feature importance
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

# 8. Save Version 2 model
joblib.dump(model, "footIQ_random_forest_v2.pkl")

print("\nModel saved as footIQ_random_forest_v2.pkl")
