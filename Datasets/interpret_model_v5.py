
import joblib
import pandas as pd
import numpy as np

# Load the saved model
bundle = joblib.load("footIQ_best_model_v5.pkl")

model = bundle["model"]
features = bundle["features"]

# Find the Logistic Regression estimator inside the pipeline
classifier = None

for step in model.named_steps.values():
    if hasattr(step, "coef_"):
        classifier = step
        break

if classifier is None:
    raise ValueError("Logistic Regression classifier not found.")

# Extract coefficients
coefficients = classifier.coef_
classes = classifier.classes_

# Map class numbers to outcome names
outcome_names = {
    0: "Home Win",
    1: "Draw",
    2: "Away Win"
}

print("--- FOOTIQ MODEL INTERPRETATION ---")
print("Model:", bundle["model_name"])
print("Features:", len(features))

# Display important features for each outcome
for index, class_value in enumerate(classes):

    outcome = outcome_names[int(class_value)]

    importance = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefficients[index]
    })

    importance["Absolute Importance"] = (
        importance["Coefficient"].abs()
    )

    importance = importance.sort_values(
        by="Absolute Importance",
        ascending=False
    )

    print("\n================================")
    print("OUTCOME:", outcome)
    print("================================")

    print("\nTop 10 influential features:")
    print(
        importance[
            ["Feature", "Coefficient"]
        ].head(10).to_string(index=False)
    )

    print("\nTop 5 positive features:")
    print(
        importance.nlargest(5, "Coefficient")[
            ["Feature", "Coefficient"]
        ].to_string(index=False)
    )

    print("\nTop 5 negative features:")
    print(
        importance.nsmallest(5, "Coefficient")[
            ["Feature", "Coefficient"]
        ].to_string(index=False)
    )
