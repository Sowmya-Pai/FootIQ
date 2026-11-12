
import re

from sql_queries import (
    team_stats,
    head_to_head,
    top_scoring_teams
)


# ---------------------------------------
# AVAILABLE EPL TEAMS
# ---------------------------------------

AVAILABLE_TEAMS = [
    "Arsenal", "Aston Villa", "Barnsley", "Birmingham",
    "Blackburn", "Blackpool", "Bolton", "Bournemouth",
    "Bradford", "Brentford", "Brighton", "Burnley",
    "Cardiff", "Charlton", "Chelsea", "Coventry",
    "Crystal Palace", "Derby", "Everton", "Fulham",
    "Huddersfield", "Hull", "Ipswich", "Leeds",
    "Leicester", "Liverpool", "Man City", "Man United",
    "Middlesbrough", "Newcastle", "Norwich",
    "Nott'm Forest", "Oldham", "Portsmouth", "QPR",
    "Reading", "Sheffield United", "Sheffield Weds",
    "Southampton", "Stoke", "Sunderland", "Swansea",
    "Swindon", "Tottenham", "Watford", "West Brom",
    "West Ham", "Wigan", "Wimbledon", "Wolves"
]


# ---------------------------------------
# 1. EXTRACT TEAM NAMES
# ---------------------------------------

def extract_teams(question):

    found_teams = []

    question_lower = question.lower()

    # Check longer names first to avoid
    # partial matches.

    sorted_teams = sorted(
        AVAILABLE_TEAMS,
        key=len,
        reverse=True
    )

    for team in sorted_teams:

        pattern = (
            r"(?<![a-z])"
            + re.escape(team.lower())
            + r"(?![a-z])"
        )

        if re.search(pattern, question_lower):

            found_teams.append(team)

    return found_teams


# ---------------------------------------
# 2. IDENTIFY QUESTION TYPE
# ---------------------------------------


def identify_question_type(question):

    question = question.lower()

    # 1. Top scoring teams
    if any(word in question for word in [
        "top scoring",
        "most goals",
        "highest scoring",
        "scored the most",
        "top goal scoring"
    ]):
        return "top_scoring"

    # 2. Explicit head-to-head questions
    if any(word in question for word in [
        "head to head",
        "head-to-head",
        "head to head record",
        "h2h"
    ]):
        return "head_to_head"

    # 3. General comparison between two teams
    if any(word in question for word in [
        "compare",
        "comparison",
        "compare the teams",
        "team comparison"
    ]):
        return "team_comparison"

    # 4. Other head-to-head questions
    if any(word in question for word in [
        "versus",
        " vs ",
        "vs."
    ]):
        return "head_to_head"

    # 5. Individual team statistics
    if any(word in question for word in [
        "statistics",
        "stats",
        "wins",
        "won",
        "draws",
        "losses",
        "lost",
        "average goals",
        "goals scored",
        "goals conceded",
        "matches played",
        "how many matches",
        "record"
    ]):
        return "team_stats"

    return "unknown"


# ---------------------------------------
# 3. PROCESS USER QUESTION
# ---------------------------------------


def process_sql_question(question):

    question_type = identify_question_type(question)

    teams = extract_teams(question)

    print("\nQuestion type:", question_type)
    print("Detected teams:", teams)

    # TOP SCORING TEAMS

    if question_type == "top_scoring":

        results = top_scoring_teams(10)

        return {
            "type": "top_scoring",
            "results": results
        }

    # HEAD-TO-HEAD

    elif question_type == "head_to_head":

        if len(teams) < 2:

            return {
                "error": "Please mention two EPL teams."
            }

        result = head_to_head(
            teams[0],
            teams[1]
        )

        if result is None:

            return {
                "error": "No historical matches found."
            }

        return {
            "type": "head_to_head",
            "results": result
        }

    # COMPARISON OF TWO TEAMS' OVERALL STATISTICS

    elif question_type == "team_comparison":

        if len(teams) < 2:

            return {
                "error": "Please mention two EPL teams to compare."
            }

        comparison_results = {}

        for team in teams[:2]:

            result = team_stats(team)

            if result is None:

                return {
                    "error": f"No historical matches found for {team}."
                }

            comparison_results[team] = result

        return {
            "type": "team_comparison",
            "results": comparison_results
        }

    # INDIVIDUAL TEAM STATISTICS

    elif question_type == "team_stats":

        if len(teams) < 1:

            return {
                "error": "Please mention an EPL team."
            }

        result = team_stats(teams[0])

        if result is None:

            return {
                "error": "No historical matches found."
            }

        return {
            "type": "team_stats",
            "results": result
        }

    # UNKNOWN QUESTION

    else:

        return {
            "error": "I couldn't identify the SQL analytics question."
        }



# ---------------------------------------
# 4. TEST THE QUESTION ROUTER
# ---------------------------------------

if __name__ == "__main__":

    print("\n================================")
    print("       FOOTIQ SQL ASSISTANT")
    print("================================")

    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit", "stop"]:
            print("FootIQ: Goodbye!")
            break

        result = process_sql_question(question)

        print("\n========== SQL RESULT ==========")

        if "error" in result:

            print("FootIQ:", result["error"])

        else:

            print("Analytics Type:", result["type"])
            print("Retrieved Data:", result["results"])
