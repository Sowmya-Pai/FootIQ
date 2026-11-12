
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

# 1. Load dataset
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

X = df[features]
y = df["Target"]

# 2. Chronological train, validation and test split
test_split = int(len(df) * 0.8)

X_train_full = X.iloc[:test_split]
y_train_full = y.iloc[:test_split]

X_test = X.iloc[test_split:]
y_test = y.iloc[test_split:]

validation_split = int(len(X_train_full) * 0.8)

X_train = X_train_full.iloc[:validation_split]
y_train = y_train_full.iloc[:validation_split]

X_val = X_train_full.iloc[validation_split:]
y_val = y_train_full.iloc[validation_split:]

# 3. Train using selected Version 3 weights
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight={0: 1, 1: 2.5, 2: 1.2},
    n_jobs=-1
)

model.fit(X_train, y_train)

# 4. Get validation probabilities
val_probabilities = model.predict_proba(X_val)

# Locate the probability columns by class label
draw_index = list(model.classes_).index(1)
home_index = list(model.classes_).index(0)
away_index = list(model.classes_).index(2)

# Default predictions: highest probability wins
default_predictions = model.predict(X_val)

print("--- DEFAULT VALIDATION PERFORMANCE ---")
print(
    "Accuracy:",
    round(accuracy_score(y_val, default_predictions) * 100, 2),
    "%"
)

print(
    "Macro F1:",
    round(f1_score(y_val, default_predictions,
                   average="macro", zero_division=0), 3)
)

# 5. Experiment with draw probability thresholds
threshold_results = []

for threshold in np.arange(0.10, 0.61, 0.01):

    predictions = []

    for probabilities in val_probabilities:

        home_probability = probabilities[home_index]
        draw_probability = probabilities[draw_index]
        away_probability = probabilities[away_index]

        if draw_probability >= threshold:
            prediction = 1

        elif home_probability >= away_probability:
            prediction = 0

        else:
            prediction = 2

        predictions.append(prediction)

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(y_val, predictions)
    accuracy = accuracy_score(y_val, predictions)

    report = classification_report(
        y_val,
        predictions,
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": round(float(threshold), 2),
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Balanced Accuracy": balanced_acc,
        "Draw Recall": report["1"]["recall"]
    })

# 6. Display top configurations
results_df = pd.DataFrame(threshold_results)

print("\n--- TOP 10 DRAW THRESHOLDS ---")

print(
    results_df.sort_values(
        by=["Macro F1", "Balanced Accuracy"],
        ascending=False
    ).head(10).to_string(index=False)
)

# Select threshold using validation macro F1
best_result = results_df.sort_values(
    by=["Macro F1", "Balanced Accuracy"],
    ascending=False
).iloc[0]

best_threshold = float(best_result["Threshold"])

print("\nSelected draw threshold:", best_threshold)

# 7. Retrain on all training data
final_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight={0: 1, 1: 2.5, 2: 1.2},
    n_jobs=-1
)

final_model.fit(X_train_full, y_train_full)

# 8. Evaluate on untouched test set
test_probabilities = final_model.predict_proba(X_test)

test_predictions = []

for probabilities in test_probabilities:

    home_probability = probabilities[home_index]
    draw_probability = probabilities[draw_index]
    away_probability = probabilities[away_index]

    if draw_probability >= best_threshold:
        prediction = 1

    elif home_probability >= away_probability:
        prediction = 0

    else:
        prediction = 2

    test_predictions.append(prediction)

print("\n--- VERSION 4 FINAL TEST PERFORMANCE ---")

print(
    "Accuracy:",
    round(accuracy_score(y_test, test_predictions) * 100, 2),
    "%"
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_test, test_predictions), 3)
)

print("\n--- CLASSIFICATION REPORT ---")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1, 2],
        target_names=["Home Win", "Draw", "Away Win"],
        zero_division=0
    )
)

print("\n--- CONFUSION MATRIX ---")

print(confusion_matrix(y_test, test_predictions, labels=[0, 1, 2]))

print("\n--- PREDICTED OUTCOME COUNTS ---")

print(pd.Series(test_predictions).value_counts().sort_index())

# 9. Save model and threshold together
joblib.dump(
    {
        "model": final_model,
        "features": features,
        "draw_threshold": best_threshold
    },
    "footIQ_random_forest_v4.pkl"
)

print("\nVersion 4 model saved successfully!")
