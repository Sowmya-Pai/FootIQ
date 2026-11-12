
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report
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

# 2. Chronological train-validation split
split_index = int(len(df) * 0.8)

X_train_full = X.iloc[:split_index]
y_train_full = y.iloc[:split_index]

validation_index = int(len(X_train_full) * 0.8)

X_train = X_train_full.iloc[:validation_index]
y_train = y_train_full.iloc[:validation_index]

X_val = X_train_full.iloc[validation_index:]
y_val = y_train_full.iloc[validation_index:]

print("--- DATA SPLIT ---")
print("Training matches:", len(X_train))
print("Validation matches:", len(X_val))

# 3. Define three algorithms
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}

results = []
trained_models = {}

# 4. Train and evaluate on validation data
for name, model in models.items():

    print(f"\nTraining {name}...")

    if name == "Hist Gradient Boosting":
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weights
        ) if isinstance(model, Pipeline) else model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )
    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_val,
        predictions
    )

    report = classification_report(
        y_val,
        predictions,
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0
    )

    draw_recall = report["1"]["recall"]

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Balanced Accuracy": balanced_acc,
        "Draw Recall": draw_recall
    })

    trained_models[name] = model

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1, 3))
    print("Balanced Accuracy:", round(balanced_acc, 3))
    print("Draw Recall:", round(draw_recall * 100, 2), "%")

# 5. Compare models
results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")

print(
    results_df.sort_values(
        by="Macro F1",
        ascending=False
    ).to_string(index=False)
)

# 6. Select the best model using validation Macro F1
best_name = results_df.sort_values(
    by=["Macro F1", "Balanced Accuracy"],
    ascending=False
).iloc[0]["Model"]

print("\nBest validation model:", best_name)

# 7. Retrain selected model on the full first 80%
best_model = models[best_name]

if best_name == "Hist Gradient Boosting":

    full_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_full
    )

    best_model.fit(
        X_train_full,
        y_train_full,
        sample_weight=full_sample_weights
    )

else:
    best_model.fit(X_train_full, y_train_full)

# 8. Save selected model
joblib.dump(
    {
        "model": best_model,
        "features": features,
        "model_name": best_name
    },
    "footIQ_best_model_v5.pkl"
)

print("\nBest model saved as footIQ_best_model_v5.pkl")
print("Validation comparison completed successfully.")
