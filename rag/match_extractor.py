
import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.5-flash-lite"


# ---------------------------------------
# AVAILABLE DATASET TEAMS
# ---------------------------------------

AVAILABLE_TEAMS = [
    "Arsenal", "Aston Villa", "Barnsley", "Birmingham",
    "Blackburn", "Blackpool", "Bolton", "Bournemouth",
    "Bradford", "Brentford", "Brighton", "Burnley",
    "Cardiff", "Charlton", "Chelsea", "Coventry",
    "Crystal Palace", "Derby", "Everton", "Fulham",
    "Huddersfield", "Hull", "Ipswich", "Leeds",
    "Leicester", "Liverpool", "Man City", "Man United",
    "Middlesbrough", "Newcastle", "Norwich", "Nott'm Forest",
    "Oldham", "Portsmouth", "QPR", "Reading",
    "Sheffield United", "Sheffield Weds", "Southampton",
    "Stoke", "Sunderland", "Swansea", "Swindon",
    "Tottenham", "Watford", "West Brom", "West Ham",
    "Wigan", "Wimbledon", "Wolves"
]


# ---------------------------------------
# EXTRACT MATCH DETAILS
# ---------------------------------------

def extract_match_details(question):

    team_list = ", ".join(AVAILABLE_TEAMS)

    prompt = f"""
    Extract the football match details from the user question.

    Available team names:
    {team_list}

    Rules:
    1. Identify the home team and away team.
    2. Match names must exactly match the available team names.
    3. The first team mentioned is normally the home team.
    4. The second team mentioned is normally the away team.
    5. If the user explicitly specifies home and away roles,
       follow those roles instead.
    6. Extract the date in YYYY-MM-DD format if provided.
    7. If no date is provided, return null.
    8. If a team cannot be identified, return null.
    9. Do not invent match details.

    Return ONLY valid JSON in this format:

    {{
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_date": "2022-04-11"
    }}

    User question:
    {question}
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=150
        )
    )

    result = json.loads(response.text)

    home = result.get("home_team")
    away = result.get("away_team")
    match_date = result.get("match_date")

    if home not in AVAILABLE_TEAMS:
        raise ValueError("Could not identify a valid home team.")

    if away not in AVAILABLE_TEAMS:
        raise ValueError("Could not identify a valid away team.")

    if home == away:
        raise ValueError("Home and away teams cannot be the same.")

    return {
        "home_team": home,
        "away_team": away,
        "match_date": match_date
    }


# ---------------------------------------
# TEST
# ---------------------------------------

if __name__ == "__main__":

    question = input("Enter your match question: ")

    details = extract_match_details(question)

    print("\nExtracted Match Details:")
    print(json.dumps(details, indent=4))
