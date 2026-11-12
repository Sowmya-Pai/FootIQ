
import os
import sqlite3


# ---------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.join(
    BASE_DIR,
    "footiq.db"
)


# ---------------------------------------
# DATABASE CONNECTION
# ---------------------------------------

def get_connection():

    return sqlite3.connect(DB_PATH)


# ---------------------------------------
# 1. TEAM STATISTICS
# ---------------------------------------

def team_stats(team):

    connection = get_connection()
    cursor = connection.cursor()

    query = """
    SELECT

        COUNT(*) AS matches_played,

        SUM(
            CASE
                WHEN (HomeTeam = ? AND FTR = 'H')
                  OR (AwayTeam = ? AND FTR = 'A')
                THEN 1 ELSE 0
            END
        ) AS wins,

        SUM(
            CASE
                WHEN FTR = 'D'
                THEN 1 ELSE 0
            END
        ) AS draws,

        SUM(
            CASE
                WHEN (HomeTeam = ? AND FTR = 'A')
                  OR (AwayTeam = ? AND FTR = 'H')
                THEN 1 ELSE 0
            END
        ) AS losses,

        ROUND(
            AVG(
                CASE
                    WHEN HomeTeam = ? THEN FTHG
                    ELSE FTAG
                END
            ), 2
        ) AS avg_goals_scored,

        ROUND(
            AVG(
                CASE
                    WHEN HomeTeam = ? THEN FTAG
                    ELSE FTHG
                END
            ), 2
        ) AS avg_goals_conceded

    FROM matches

    WHERE HomeTeam = ? COLLATE NOCASE
       OR AwayTeam = ? COLLATE NOCASE
    """

    cursor.execute(
        query,
        (
            team, team,
            team, team,
            team, team,
            team, team
        )
    )

    result = cursor.fetchone()

    connection.close()

    if result[0] == 0:
        return None

    return {
        "team": team,
        "matches_played": result[0],
        "wins": result[1],
        "draws": result[2],
        "losses": result[3],
        "avg_goals_scored": result[4],
        "avg_goals_conceded": result[5]
    }


# ---------------------------------------
# 2. HEAD-TO-HEAD STATISTICS
# ---------------------------------------


def head_to_head(team1, team2):

    connection = get_connection()
    cursor = connection.cursor()

    query = """
    SELECT

        COUNT(*) AS total_matches,

        SUM(
            CASE
                WHEN (HomeTeam = ? COLLATE NOCASE AND FTR = 'H')
                  OR (AwayTeam = ? COLLATE NOCASE AND FTR = 'A')
                THEN 1 ELSE 0
            END
        ) AS team1_wins,

        SUM(
            CASE
                WHEN FTR = 'D'
                THEN 1 ELSE 0
            END
        ) AS draws,

        SUM(
            CASE
                WHEN (HomeTeam = ? COLLATE NOCASE AND FTR = 'H')
                  OR (AwayTeam = ? COLLATE NOCASE AND FTR = 'A')
                THEN 1 ELSE 0
            END
        ) AS team2_wins

    FROM matches

    WHERE
        (HomeTeam = ? COLLATE NOCASE AND AwayTeam = ? COLLATE NOCASE)
        OR
        (HomeTeam = ? COLLATE NOCASE AND AwayTeam = ? COLLATE NOCASE)
    """

    cursor.execute(
        query,
        (
            team1, team1,
            team2, team2,
            team1, team2,
            team2, team1
        )
    )

    result = cursor.fetchone()

    connection.close()

    if result[0] == 0:
        return None

    return {
        "team1": team1,
        "team2": team2,
        "total_matches": result[0],
        "team1_wins": result[1],
        "draws": result[2],
        "team2_wins": result[3]
    }



# ---------------------------------------
# 3. TOP SCORING TEAMS
# ---------------------------------------

def top_scoring_teams(limit=10):

    connection = get_connection()
    cursor = connection.cursor()

    query = """
    SELECT

        team,

        SUM(goals_scored) AS total_goals,

        COUNT(*) AS matches_played

    FROM (

        SELECT
            HomeTeam AS team,
            FTHG AS goals_scored
        FROM matches

        UNION ALL

        SELECT
            AwayTeam AS team,
            FTAG AS goals_scored
        FROM matches

    )

    GROUP BY team

    ORDER BY total_goals DESC

    LIMIT ?
    """

    cursor.execute(query, (limit,))

    results = cursor.fetchall()

    connection.close()

    return results


# ---------------------------------------
# 4. TEST THE SQL ENGINE
# ---------------------------------------

if __name__ == "__main__":

    print("\n========== FOOTIQ SQL ANALYTICS ==========")

    # Team statistics
    print("\n--- ARSENAL STATISTICS ---")

    stats = team_stats("Arsenal")

    if stats:
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print("Team not found.")

    # Head-to-head
    print("\n--- ARSENAL VS CHELSEA ---")

    h2h = head_to_head("Arsenal", "Chelsea")

    if h2h:
        for key, value in h2h.items():
            print(f"{key}: {value}")
    else:
        print("No matches found.")

    # Top scoring teams
    print("\n--- TOP 10 SCORING TEAMS ---")

    for team in top_scoring_teams(10):
        print(
            f"{team[0]}: {team[1]} goals "
            f"in {team[2]} team appearances"
        )

    print("\nSQL analytics test completed!")
