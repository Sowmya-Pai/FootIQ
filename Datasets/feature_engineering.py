
import pandas as pd

# 1. Load cleaned dataset
df = pd.read_csv("matches_cleaned.csv", encoding="cp1252")

# Convert date into datetime format
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)

# Sort matches chronologically
df = df.sort_values("DateTime").reset_index(drop=True)

# 2. Store historical performance for every team
team_history = {}

feature_rows = []

# 3. Process matches date by date
for match_date, day_matches in df.groupby(
    df["DateTime"].dt.date,
    sort=True
):

    pending_updates = []

    # Calculate features before updating team history
    for _, match in day_matches.iterrows():

        home = match["HomeTeam"]
        away = match["AwayTeam"]

        home_goals = match["FTHG"]
        away_goals = match["FTAG"]

        # Retrieve previous matches
        home_history = team_history.get(home, [])
        away_history = team_history.get(away, [])

        # Consider only the previous five matches
        home_last5 = home_history[-5:]
        away_last5 = away_history[-5:]

        # Generate historical features
        if home_last5 and away_last5:

            features = {
                "home_avg_points": sum(
                    x["points"] for x in home_last5
                ) / len(home_last5),

                "away_avg_points": sum(
                    x["points"] for x in away_last5
                ) / len(away_last5),

                "home_win_rate": sum(
                    x["win"] for x in home_last5
                ) / len(home_last5),

                "away_win_rate": sum(
                    x["win"] for x in away_last5
                ) / len(away_last5),

                "home_avg_goals_scored": sum(
                    x["goals_for"] for x in home_last5
                ) / len(home_last5),

                "home_avg_goals_conceded": sum(
                    x["goals_against"] for x in home_last5
                ) / len(home_last5),

                "away_avg_goals_scored": sum(
                    x["goals_for"] for x in away_last5
                ) / len(away_last5),

                "away_avg_goals_conceded": sum(
                    x["goals_against"] for x in away_last5
                ) / len(away_last5),

                "home_previous_matches": len(home_history),
                "away_previous_matches": len(away_history)
            }

        else:
            # Insufficient historical information
            features = {
                "home_avg_points": None,
                "away_avg_points": None,
                "home_win_rate": None,
                "away_win_rate": None,
                "home_avg_goals_scored": None,
                "home_avg_goals_conceded": None,
                "away_avg_goals_scored": None,
                "away_avg_goals_conceded": None,
                "home_previous_matches": len(home_history),
                "away_previous_matches": len(away_history)
            }

        feature_rows.append(features)

        # Prepare match information for history update
        if home_goals > away_goals:
            home_points, away_points = 3, 0
            home_win, away_win = 1, 0

        elif home_goals < away_goals:
            home_points, away_points = 0, 3
            home_win, away_win = 0, 1

        else:
            home_points, away_points = 1, 1
            home_win, away_win = 0, 0

        pending_updates.append(
            (home, {
                "points": home_points,
                "win": home_win,
                "goals_for": home_goals,
                "goals_against": away_goals
            })
        )

        pending_updates.append(
            (away, {
                "points": away_points,
                "win": away_win,
                "goals_for": away_goals,
                "goals_against": home_goals
            })
        )

    # Update histories only after processing the entire date
    for team, result in pending_updates:
        team_history.setdefault(team, []).append(result)

# 4. Add features to the original dataset
features_df = pd.DataFrame(feature_rows)

df = pd.concat([df, features_df], axis=1)

# 5. Remove matches without enough historical information
feature_columns = features_df.columns.tolist()

df = df.dropna(subset=feature_columns).reset_index(drop=True)

# 6. Save the feature-engineered dataset
df.to_csv("matches_with_features.csv", index=False)

# 7. Display results
print("\n--- FEATURE ENGINEERING COMPLETE ---")

print("Original matches: 11113")
print("Matches after feature engineering:", len(df))

print("\nNew feature columns:")
print(feature_columns)

print("\nFeature dataset preview:")
print(df[[
    "HomeTeam",
    "AwayTeam",
    "home_avg_points",
    "away_avg_points",
    "home_avg_goals_scored",
    "away_avg_goals_scored",
    "FTR"
]].head(10))

print("\nSaved as matches_with_features.csv")
