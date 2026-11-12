
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

# 1. Load Version 2 advanced dataset
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

# 2. First split: reserve the final 20% as untouched test data
test_split = int(len(df) * 0.8)

X_train_full = X.iloc[:test_split]
y_train_full = y.iloc[:test_split]

X_test = X.iloc[test_split:]
y_test = y.iloc[test_split:]

# 3. Second chronological split: training and validation
validation_split = int(len(X_train_full) * 0.8)

X_train = X_train_full.iloc[:validation_split]
y_train = y_train_full.iloc[:validation_split]

X_val = X_train_full.iloc[validation_split:]
y_val = y_train_full.iloc[validation_split:]

print("--- DATA SPLIT ---")
print("Training matches:", len(X_train))
print("Validation matches:", len(X_val))
print("Final testing matches:", len(X_test))

# 4. Experiment with different class weights
weight_options = {
    "Balanced": "balanced",
    "Draw weight 1.5": {0: 1, 1: 1.5, 2: 1},
    "Draw weight 2": {0: 1, 1: 2, 2: 1},
    "Draw weight 3": {0: 1, 1: 3, 2: 1},
    "Draw weight 2.5, Away 1.2": {0: 1, 1: 2.5, 2: 1.2}
}

results = []

print("\n--- VALIDATION EXPERIMENTS ---")

for name, weights in weight_options.items():

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight=weights,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, val_predictions)
    macro_f1 = f1_score(
        y_val, val_predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )
    balanced_acc = balanced_accuracy_score(
        y_val, val_predictions
    )

    report = classification_report(
        y_val,
        val_predictions,
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0
    )

    draw_recall = report["1"]["recall"]

    results.append({
        "Name": name,
        "Weights": weights,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Balanced Accuracy": balanced_acc,
        "Draw Recall": draw_recall
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Balanced Accuracy: {balanced_acc:.3f}")
    print(f"Draw Recall: {draw_recall * 100:.2f}%")

# 5. Select the best configuration using validation macro F1
results_df = pd.DataFrame(results)

print("\n--- VALIDATION COMPARISON ---")
print(
    results_df[
        ["Name", "Accuracy", "Macro F1",
         "Balanced Accuracy", "Draw Recall"]
    ].to_string(index=False)
)

best_result = max(
    results,
    key=lambda result: (
        result["Macro F1"],
        result["Draw Recall"]
    )
)

print("\n--- SELECTED CONFIGURATION ---")
print("Best setting:", best_result["Name"])
print("Class weights:", best_result["Weights"])

# 6. Retrain selected model on the complete training portion
final_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight=best_result["Weights"],
    n_jobs=-1
)

final_model.fit(X_train_full, y_train_full)

# 7. Evaluate once on the untouched test set
test_predictions = final_model.predict(X_test)

print("\n--- FINAL TEST PERFORMANCE ---")
print(
    f"Accuracy: {accuracy_score(y_test, test_predictions) * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy_score(y_test, test_predictions):.3f}"
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

# 8. Save Version 3
joblib.dump(final_model, "footIQ_random_forest_v3.pkl")

print("\nModel saved as footIQ_random_forest_v3.pkl")
