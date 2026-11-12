
import pandas as pd

df = pd.read_csv("matches_with_features.csv")

feature_columns = [
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

print("--- FEATURE VALIDATION ---")

print("\nMissing values:")
print(df[feature_columns].isnull().sum())

print("\nFeature statistics:")
print(df[feature_columns].describe().round(2))

print("\nTarget distribution:")
print(df["Target"].value_counts())

print("\nDate order:")
print("Chronologically sorted:",
      pd.to_datetime(df["DateTime"], utc=True).is_monotonic_increasing)

print("\nDuplicate rows:", df.duplicated().sum())
