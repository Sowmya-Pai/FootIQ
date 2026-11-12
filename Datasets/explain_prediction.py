
import pandas as pd
import numpy as np
import json

from predict_model import (
    model,
    features,
    df,
    build_history,
    calculate_features
)

OUTCOME_NAMES = {
    0: "Home Win",
    1: "Draw",
    2: "Away Win"
}

# ---------------------------------------
# FEATURE DEFINITIONS
# ---------------------------------------

FEATURE_DEFINITIONS = {

    "home_avg_points":
        "Average points earned by the home team in its last 5 matches.",

    "away_avg_points":
        "Average points earned by the away team in its last 5 matches.",

    "home_win_rate":
        "Proportion of the home team's last 5 matches that it won.",

    "away_win_rate":
        "Proportion of the away team's last 5 matches that it won.",

    "home_avg_goals_scored":
        "Average goals scored by the home team in its last 5 matches.",

    "home_avg_goals_conceded":
        "Average goals conceded by the home team in its last 5 matches.",

    "away_avg_goals_scored":
        "Average goals scored by the away team in its last 5 matches.",

    "away_avg_goals_conceded":
        "Average goals conceded by the away team in its last 5 matches.",

    "home_previous_matches":
        "Number of historical matches involving the home team before the prediction date.",

    "away_previous_matches":
        "Number of historical matches involving the away team before the prediction date.",

    "home_home_avg_points":
        "Average points earned by the home team in its last 5 home matches.",

    "away_away_avg_points":
        "Average points earned by the away team in its last 5 away matches.",

    "home_home_avg_goal_diff":
        "Average goal difference for the home team in its last 5 home matches.",

    "away_away_avg_goal_diff":
        "Average goal difference for the away team in its last 5 away matches.",

    "h2h_home_avg_points":
        "Average points earned by the home team in its last 5 head-to-head matches against the away team.",

    "h2h_home_avg_goal_diff":
        "Average goal difference for the home team in its last 5 head-to-head matches against the away team.",

    "elo_difference":
        "Home team's Elo rating minus the away team's Elo rating. A negative value means the away team has the higher Elo rating."
}


def explain_match(home, away, prediction_date):

    if home == away:
        raise ValueError("Home and away teams must be different.")

    available_teams = set(df["HomeTeam"]) | set(df["AwayTeam"])

    if home not in available_teams:
        raise ValueError(f"Unknown home team: {home}")

    if away not in available_teams:
        raise ValueError(f"Unknown away team: {away}")

    # Rebuild historical information
    build_history(prediction_date)

    # Calculate match features
    match_features = calculate_features(home, away)

    input_data = pd.DataFrame(
        [[match_features[f] for f in features]],
        columns=features
    )

    # Access pipeline components
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]

    # Scale input features
    scaled_features = scaler.transform(input_data)

    # Generate prediction
    probabilities = model.predict_proba(input_data)[0]
    predicted_class = int(model.predict(input_data)[0])

    # Calculate feature contributions
    contributions = scaled_features[0] * classifier.coef_

    class_scores = (
        contributions.sum(axis=1) + classifier.intercept_
    )

    # Prepare structured explanations
    class_explanations = {}

    for i, class_value in enumerate(model.classes_):

        outcome = OUTCOME_NAMES[int(class_value)]

        contribution_data = pd.DataFrame({
            "feature": features,
            "contribution": contributions[i]
        })

        positive = contribution_data[
            contribution_data["contribution"] > 0
        ].sort_values(
            "contribution",
            ascending=False
        ).head(5)

        negative = contribution_data[
            contribution_data["contribution"] < 0
        ].sort_values(
            "contribution",
            ascending=True
        ).head(5)

        class_explanations[outcome] = {
            "class_score": float(class_scores[i]),

            "positive_factors": [
                {
                    "feature": row["feature"],
                    "meaning": FEATURE_DEFINITIONS[row["feature"]],
                    "value": float(match_features[row["feature"]]),
                    "contribution": float(row["contribution"])
}
                for _, row in positive.iterrows()
            ],

            "negative_factors": [
                {
                    "feature": row["feature"],
                    "meaning": FEATURE_DEFINITIONS[row["feature"]],
                    "value": float(match_features[row["feature"]]),
                    "contribution": float(row["contribution"])
                }
                for _, row in negative.iterrows()
            ]
        }

    # Final structured output
    result = {
        "home_team": home,
        "away_team": away,
        "match_date": prediction_date,

        "predicted_outcome": OUTCOME_NAMES[predicted_class],

        "probabilities": {
            OUTCOME_NAMES[int(class_value)]:
                float(probability * 100)
            for class_value, probability
            in zip(model.classes_, probabilities)
        },

        "match_features": {
            key: float(value)
            for key, value in match_features.items()
        },

        "class_explanations": class_explanations
    }

    return result


if __name__ == "__main__":

    explanation = explain_match(
        "Arsenal",
        "Chelsea",
        "2022-04-11"
    )

    print("\n========== FOOTIQ STRUCTURED ML OUTPUT ==========")

    print(json.dumps(explanation, indent=4))

    print("\nExplanation data generated successfully!")
