
import pandas as pd
import numpy as np

from predict_model import (
    features,
    build_history,
    calculate_features
)

feature_df = pd.read_csv(
    "matches_with_advanced_features.csv",
    encoding="cp1252"
)

feature_df["DateTime"] = pd.to_datetime(
    feature_df["DateTime"], utc=True
)

feature_df = feature_df.sort_values(
    "DateTime"
).reset_index(drop=True)

# Select 10 matches spread across the dataset
sample_indices = np.linspace(
    0, len(feature_df) - 1, 10, dtype=int
)

total_checks = 0
total_matches = 0
failed_matches = []

print("FOOTIQ MULTI-MATCH VALIDATION")
print("=" * 45)

for index in sample_indices:

    sample = feature_df.iloc[index]

    home = sample["HomeTeam"]
    away = sample["AwayTeam"]
    match_date = sample["DateTime"].strftime("%Y-%m-%d")

    build_history(match_date)

    calculated = calculate_features(home, away)

    mismatches = []

    for feature in features:

        stored = float(sample[feature])
        predicted = float(calculated[feature])

        total_checks += 1

        if np.isclose(stored, predicted, atol=1e-6):
            total_matches += 1
        else:
            mismatches.append(feature)

    if mismatches:
        failed_matches.append((home, away, match_date, mismatches))
        status = "FAILED"
    else:
        status = "PASSED"

    print(
        f"{home} vs {away} | {match_date} | {status}"
    )

print("\n" + "=" * 45)
print("FINAL VALIDATION RESULTS")
print("=" * 45)

print("Matches tested:", len(sample_indices))
print("Features checked:", total_checks)
print("Features matched:", total_matches)
print("Features mismatched:", total_checks - total_matches)

if not failed_matches:
    print("\nALL 10 MATCHES PASSED!")
else:
    print("\nMISMATCHES FOUND:")
    for item in failed_matches:
        print(item)
