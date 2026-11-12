
import pandas as pd
import numpy as np

# 1. Load the cleaned dataset
df = pd.read_csv("matches_cleaned.csv", encoding="cp1252")

df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
df = df.sort_values("DateTime").reset_index(drop=True)

# Historical information stored for each team
team_history = {}
home_venue_history = {}
away_venue_history = {}

# Head-to-head history for each ordered pair of teams
h2h_history = {}

# Initial Elo rating for every team
elo_ratings = {}

feature_rows = []

# Features that need at least one previous match
base_features = [
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

# 2. Process matches date by date
for match_date, daily_matches in df.groupby(df["DateTime"].dt.date):

    pending_updates = []

    # All matches on this date use information from before this date
    for index, row in daily_matches.iterrows():

        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_goals = int(row["FTHG"])
        away_goals = int(row["FTAG"])

        # Target: 0 = Home Win, 1 = Draw, 2 = Away Win
        result = int(row["Target"])

        if result == 0:
            home_points, away_points = 3, 0
        elif result == 1:
            home_points, away_points = 1, 1
        else:
            home_points, away_points = 0, 3

        home_history = team_history.get(home, [])
        away_history = team_history.get(away, [])

        # Skip matches where either team has no previous match history
        if not home_history or not away_history:
            home_elo = elo_ratings.get(home, 1500)
            away_elo = elo_ratings.get(away, 1500)
            pending_updates.append({
                "home": home,
                "away": away,
                "home_points": home_points,
                "away_points": away_points,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "home_elo": home_elo,
                "away_elo": away_elo
            })

            continue

        # 3. Overall recent form: previous five matches
        home_recent = home_history[-5:]
        away_recent = away_history[-5:]

        features = {
            "home_avg_points": np.mean([m["points"] for m in home_recent]),
            "away_avg_points": np.mean([m["points"] for m in away_recent]),

            "home_win_rate": np.mean([m["points"] == 3 for m in home_recent]),
            "away_win_rate": np.mean([m["points"] == 3 for m in away_recent]),

            "home_avg_goals_scored": np.mean([m["scored"] for m in home_recent]),
            "home_avg_goals_conceded": np.mean([m["conceded"] for m in home_recent]),

            "away_avg_goals_scored": np.mean([m["scored"] for m in away_recent]),
            "away_avg_goals_conceded": np.mean([m["conceded"] for m in away_recent]),

            "home_previous_matches": len(home_history),
            "away_previous_matches": len(away_history)
        }

        # 4. Venue-specific form
        home_venue = home_venue_history.get(home, [])[-5:]
        away_venue = away_venue_history.get(away, [])[-5:]

        features["home_home_avg_points"] = (
            np.mean([m["points"] for m in home_venue])
            if home_venue else 1.0
        )

        features["away_away_avg_points"] = (
            np.mean([m["points"] for m in away_venue])
            if away_venue else 1.0
        )

        features["home_home_avg_goal_diff"] = (
            np.mean([m["goal_diff"] for m in home_venue])
            if home_venue else 0.0
        )

        features["away_away_avg_goal_diff"] = (
            np.mean([m["goal_diff"] for m in away_venue])
            if away_venue else 0.0
        )

        # 5. Head-to-head history
        previous_h2h = h2h_history.get((home, away), [])[-5:]

        features["h2h_home_avg_points"] = (
            np.mean([m["points"] for m in previous_h2h])
            if previous_h2h else 1.0
        )

        features["h2h_home_avg_goal_diff"] = (
            np.mean([m["goal_diff"] for m in previous_h2h])
            if previous_h2h else 0.0
        )

        # 6. Elo rating difference
        home_elo = elo_ratings.get(home, 1500)
        away_elo = elo_ratings.get(away, 1500)

        features["elo_difference"] = home_elo - away_elo

        feature_rows.append({"index": index, **features})

        # Store match details for updating histories after this date
        pending_updates.append({
            "home": home,
            "away": away,
            "home_points": home_points,
            "away_points": away_points,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
            "home_elo": home_elo,
            "away_elo": away_elo
        })

    # 7. Update team histories only after processing every match that day
    for match in pending_updates:

        home = match["home"]
        away = match["away"]

        home_points = match["home_points"]
        away_points = match["away_points"]

        home_goals = match["home_goals"]
        away_goals = match["away_goals"]

        team_history.setdefault(home, []).append({
            "points": home_points,
            "scored": home_goals,
            "conceded": away_goals
        })

        team_history.setdefault(away, []).append({
            "points": away_points,
            "scored": away_goals,
            "conceded": home_goals
        })

        # Home team's previous home performances
        home_venue_history.setdefault(home, []).append({
            "points": home_points,
            "goal_diff": home_goals - away_goals
        })

        # Away team's previous away performances
        away_venue_history.setdefault(away, []).append({
            "points": away_points,
            "goal_diff": away_goals - home_goals
        })

        # Store head-to-head from both teams' perspectives
        h2h_history.setdefault((home, away), []).append({
            "points": home_points,
            "goal_diff": home_goals - away_goals
        })

        h2h_history.setdefault((away, home), []).append({
            "points": away_points,
            "goal_diff": away_goals - home_goals
        })

        # Update Elo ratings after the match
        expected_home = 1 / (
            1 + 10 ** ((match["away_elo"] - match["home_elo"] - 100) / 400)
        )

        if match["result"] == 0:
            actual_home = 1
        elif match["result"] == 1:
            actual_home = 0.5
        else:
            actual_home = 0

        k_factor = 20

        change = k_factor * (actual_home - expected_home)

        elo_ratings[home] = match["home_elo"] + change
        elo_ratings[away] = match["away_elo"] - change


# 8. Combine the new features with the original dataset
features_df = pd.DataFrame(feature_rows).set_index("index")

result_df = df.join(features_df)

# Remove rows without the required previous-match history
result_df = result_df.dropna(subset=base_features)

# Save the new dataset
result_df.to_csv("matches_with_advanced_features.csv", index=False)

print("\n--- ADVANCED FEATURE ENGINEERING COMPLETE ---")
print("Original matches:", len(df))
print("Matches retained:", len(result_df))
print("New feature count:", len(features_df.columns))

print("\nNew features:")
print([
    "home_home_avg_points",
    "away_away_avg_points",
    "home_home_avg_goal_diff",
    "away_away_avg_goal_diff",
    "h2h_home_avg_points",
    "h2h_home_avg_goal_diff",
    "elo_difference"
])

print("\nMissing values in new features:")
print(result_df[features_df.columns].isnull().sum())

print("\nFeature preview:")
print(result_df[
    ["HomeTeam", "AwayTeam", "FTR",
     "home_home_avg_points", "away_away_avg_points",
     "h2h_home_avg_points", "elo_difference"]
].head(10))

print("\nSaved as matches_with_advanced_features.csv")
