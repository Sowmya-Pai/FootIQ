
from flask import Flask, render_template, request, jsonify
import sys
import os

# --------------------------------------------------
# 1. INITIALIZE FLASK
# --------------------------------------------------

app = Flask(__name__)

# --------------------------------------------------
# 2. CONNECT TO EXISTING ML MODEL
# --------------------------------------------------

# Add Datasets folder to Python's search path
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Datasets"
)

sys.path.insert(0, DATASET_PATH)

from predict_model import predict_match, df
# Connect FootIQ RAG assistant
RAG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rag"
)

sys.path.insert(0, RAG_PATH)

from rag_chat import process_question
# --------------------------------------------------
# 3. HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    teams = sorted(
        set(df["HomeTeam"].unique()) |
        set(df["AwayTeam"].unique())
    )

    return render_template(
        "index.html",
        teams=teams
    )


# --------------------------------------------------
# 4. PREDICTION API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        home_team = data.get("home_team")
        away_team = data.get("away_team")
        match_date = data.get("match_date")

        if not home_team or not away_team or not match_date:
            return jsonify({
                "error": "Please provide both teams and match date."
            }), 400

        if home_team == away_team:
            return jsonify({
                "error": "Home and away teams must be different."
            }), 400

        # Generate historical features
        from predict_model import build_history, calculate_features

        build_history(match_date)

        match_features = calculate_features(
            home_team,
            away_team
        )

        # Load model and feature order from existing bundle
        import joblib

        model_bundle = joblib.load(
            os.path.join(DATASET_PATH, "footIQ_best_model_v5.pkl")
        )

        model = model_bundle["model"]
        features = model_bundle["features"]

        import pandas as pd

        input_data = pd.DataFrame(
            [[match_features[f] for f in features]],
            columns=features
        )

        probabilities = model.predict_proba(input_data)[0]
        prediction = int(model.predict(input_data)[0])

        outcome_names = {
            0: "Home Win",
            1: "Draw",
            2: "Away Win"
        }

        results = {
            outcome_names[int(cls)]: round(float(prob) * 100, 2)
            for cls, prob in zip(model.classes_, probabilities)
        }

        return jsonify({
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            "predicted_outcome": outcome_names[prediction],
            "probabilities": results,
            "elo_difference": round(
                match_features["elo_difference"], 2
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# --------------------------------------------------
# 5. FOOTIQ AI CHAT API
# --------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        question = data.get("question", "").strip()

        if not question:

            return jsonify({
                "error": "Please enter a question."
            }), 400

        # Process question using RAG, ML and SQL
        result = process_question(question)

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# --------------------------------------------------
# 5. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
