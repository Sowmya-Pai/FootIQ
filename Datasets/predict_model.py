
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --------------------------------------------------
# 1. LOAD DATA AND MODEL
# --------------------------------------------------

df = pd.read_csv(os.path.join(BASE_DIR, "matches_cleaned.csv"), encoding="cp1252")

df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
df = df.sort_values("DateTime").reset_index(drop=True)

bundle = joblib.load(os.path.join(BASE_DIR, "footIQ_best_model_v5.pkl"))

model = bundle["model"]
features = bundle["features"]

# --------------------------------------------------
# 2. BUILD HISTORICAL TEAM INFORMATION
# --------------------------------------------------

team_history = {}
home_venue_history = {}
away_venue_history = {}
h2h_history = {}
elo_ratings = {}


def update_histories(home, away, home_goals, away_goals, result,
                     home_elo, away_elo):

    if result == 0:
        home_points, away_points = 3, 0
        actual_home = 1
    elif result == 1:
        home_points, away_points = 1, 1
        actual_home = 0.5
    else:
        home_points, away_points = 0, 3
        actual_home = 0

    # Overall team history
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

    # Venue-specific history
    home_venue_history.setdefault(home, []).append({
        "points": home_points,
        "goal_diff": home_goals - away_goals
    })

    away_venue_history.setdefault(away, []).append({
        "points": away_points,
        "goal_diff": away_goals - home_goals
    })

    # Head-to-head history
    h2h_history.setdefault((home, away), []).append({
        "points": home_points,
        "goal_diff": home_goals - away_goals
    })

    h2h_history.setdefault((away, home), []).append({
        "points": away_points,
        "goal_diff": away_goals - home_goals
    })

    # Elo rating update
    expected_home = 1 / (
        1 + 10 ** ((away_elo - home_elo - 100) / 400)
    )

    k_factor = 20
    change = k_factor * (actual_home - expected_home)

    elo_ratings[home] = home_elo + change
    elo_ratings[away] = away_elo - change


# --------------------------------------------------
# 3. REBUILD HISTORY BEFORE THE PREDICTION DATE
# --------------------------------------------------

def build_history(prediction_date):

    global team_history, home_venue_history
    global away_venue_history, h2h_history, elo_ratings

    team_history = {}
    home_venue_history = {}
    away_venue_history = {}
    h2h_history = {}
    elo_ratings = {}

    prediction_day = pd.Timestamp(prediction_date).date()

    historical_data = df[
        df["DateTime"].dt.date < prediction_day
    ]

    # Process matches date by date
    for match_date, daily_matches in historical_data.groupby(
        historical_data["DateTime"].dt.date
    ):

        pending_updates = []

        # Capture information before processing the date
        for _, row in daily_matches.iterrows():

            home = row["HomeTeam"]
            away = row["AwayTeam"]

            home_elo = elo_ratings.get(home, 1500)
            away_elo = elo_ratings.get(away, 1500)

            pending_updates.append({
                "home": home,
                "away": away,
                "home_goals": int(row["FTHG"]),
                "away_goals": int(row["FTAG"]),
                "result": int(row["Target"]),
                "home_elo": home_elo,
                "away_elo": away_elo
            })

        # Update histories after processing the entire date
        for match in pending_updates:

            update_histories(
                match["home"],
                match["away"],
                match["home_goals"],
                match["away_goals"],
                match["result"],
                match["home_elo"],
                match["away_elo"]
            )


# --------------------------------------------------
# 4. CALCULATE THE 17 PRE-MATCH FEATURES
# --------------------------------------------------

def calculate_features(home, away):

    home_history = team_history.get(home, [])
    away_history = team_history.get(away, [])

    if not home_history or not away_history:
        raise ValueError(
            "Both teams must have at least one previous match."
        )

    home_recent = home_history[-5:]
    away_recent = away_history[-5:]

    result = {

        "home_avg_points": np.mean(
            [m["points"] for m in home_recent]
        ),

        "away_avg_points": np.mean(
            [m["points"] for m in away_recent]
        ),

        "home_win_rate": np.mean(
            [m["points"] == 3 for m in home_recent]
        ),

        "away_win_rate": np.mean(
            [m["points"] == 3 for m in away_recent]
        ),

        "home_avg_goals_scored": np.mean(
            [m["scored"] for m in home_recent]
        ),

        "home_avg_goals_conceded": np.mean(
            [m["conceded"] for m in home_recent]
        ),

        "away_avg_goals_scored": np.mean(
            [m["scored"] for m in away_recent]
        ),

        "away_avg_goals_conceded": np.mean(
            [m["conceded"] for m in away_recent]
        ),

        "home_previous_matches": len(home_history),

        "away_previous_matches": len(away_history)
    }

    # Venue-specific features
    home_venue = home_venue_history.get(home, [])[-5:]
    away_venue = away_venue_history.get(away, [])[-5:]

    result["home_home_avg_points"] = (
        np.mean([m["points"] for m in home_venue])
        if home_venue else 1.0
    )

    result["away_away_avg_points"] = (
        np.mean([m["points"] for m in away_venue])
        if away_venue else 1.0
    )

    result["home_home_avg_goal_diff"] = (
        np.mean([m["goal_diff"] for m in home_venue])
        if home_venue else 0.0
    )

    result["away_away_avg_goal_diff"] = (
        np.mean([m["goal_diff"] for m in away_venue])
        if away_venue else 0.0
    )

    # Head-to-head features
    previous_h2h = h2h_history.get((home, away), [])[-5:]

    result["h2h_home_avg_points"] = (
        np.mean([m["points"] for m in previous_h2h])
        if previous_h2h else 1.0
    )

    result["h2h_home_avg_goal_diff"] = (
        np.mean([m["goal_diff"] for m in previous_h2h])
        if previous_h2h else 0.0
    )

    # Elo difference
    home_elo = elo_ratings.get(home, 1500)
    away_elo = elo_ratings.get(away, 1500)

    result["elo_difference"] = home_elo - away_elo

    return result


# --------------------------------------------------
# 5. PREDICT MATCH OUTCOME
# --------------------------------------------------

def predict_match(home, away, prediction_date):

    if home == away:
        raise ValueError("Home and away teams must be different.")

    available_teams = set(df["HomeTeam"]) | set(df["AwayTeam"])

    if home not in available_teams:
        raise ValueError(f"Unknown home team: {home}")

    if away not in available_teams:
        raise ValueError(f"Unknown away team: {away}")

    build_history(prediction_date)

    match_features = calculate_features(home, away)

    # Keep exactly the same feature order as training
    input_data = pd.DataFrame(
        [[match_features[f] for f in features]],
        columns=features
    )

    probabilities = model.predict_proba(input_data)[0]

    outcome_names = {
        0: "Home Win",
        1: "Draw",
        2: "Away Win"
    }

    predicted_class = int(model.predict(input_data)[0])

    print("\n===================================")
    print("          FOOTIQ PREDICTION")
    print("===================================")

    print(f"Home Team: {home}")
    print(f"Away Team: {away}")
    print(f"Match Date: {prediction_date}")

    print("\n--- OUTCOME PROBABILITIES ---")

    for class_value, probability in zip(
        model.classes_, probabilities
    ):
        outcome = outcome_names[int(class_value)]

        print(f"{outcome}: {probability * 100:.2f}%")

    print("\nPredicted Outcome:",
          outcome_names[predicted_class])

    print("\n--- IMPORTANT MATCH FEATURES ---")

    print(f"Home Elo: {elo_ratings.get(home, 1500):.2f}")
    print(f"Away Elo: {elo_ratings.get(away, 1500):.2f}")
    print(f"Elo Difference: {match_features['elo_difference']:.2f}")

    print("\nPrediction completed successfully.")


# --------------------------------------------------
# 6. TEST THE PREDICTION ENGINE
# --------------------------------------------------

if __name__ == "__main__":

    HOME_TEAM = "Arsenal"
    AWAY_TEAM = "Chelsea"

    # Dataset ends on 10 April 2022
    MATCH_DATE = "2022-04-11"

    predict_match(
        HOME_TEAM,
        AWAY_TEAM,
        MATCH_DATE
    )
