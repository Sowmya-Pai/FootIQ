

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score
)

# Load the dataset
df = pd.read_csv("matches_with_advanced_features.csv")

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

# Prepare chronological test data
split_index = int(len(df) * 0.80)

test_data = df.iloc[split_index:]

X_test = test_data[features]
y_test = test_data["Target"]

# Load the selected V5 model
bundle = joblib.load("footIQ_best_model_v5.pkl")

model = bundle["model"]

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("--- VERSION 5 FINAL TEST RESULTS ---")

print("Model:", bundle["model_name"])
print("Test matches:", len(X_test))

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print(
    "Macro F1:",
    round(f1_score(y_test, y_pred, average="macro"), 3)
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_test, y_pred), 3)
)

print("\n--- CLASSIFICATION REPORT ---")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Home Win", "Draw", "Away Win"],
        zero_division=0
    )
)

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
