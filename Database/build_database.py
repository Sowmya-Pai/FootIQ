
import os
import sqlite3
import pandas as pd

# ---------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "Datasets",
    "matches_cleaned.csv"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "Database",
    "footiq.db"
)


# ---------------------------------------
# 2. LOAD DATASET
# ---------------------------------------

print("Loading football dataset...")

df = pd.read_csv(
    CSV_PATH,
    encoding="cp1252"
)

print("Total records:", len(df))


# ---------------------------------------
# 3. CLEAN DATA
# ---------------------------------------

df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    utc=True,
    errors="coerce"
)

df = df.dropna(
    subset=[
        "DateTime",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR"
    ]
)

df["DateTime"] = df["DateTime"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

# ---------------------------------------
# 4. CREATE SQLITE DATABASE
# ---------------------------------------

connection = sqlite3.connect(DB_PATH)

print("\nCreating SQLite database...")


# ---------------------------------------
# 5. STORE MATCH DATA
# ---------------------------------------

df.to_sql(
    "matches",
    connection,
    if_exists="replace",
    index=False
)


# ---------------------------------------
# 6. CREATE INDEXES
# ---------------------------------------

cursor = connection.cursor()

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_home_team
ON matches(HomeTeam)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_away_team
ON matches(AwayTeam)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_match_date
ON matches(DateTime)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_teams
ON matches(HomeTeam, AwayTeam)
""")


# ---------------------------------------
# 7. VERIFY DATABASE
# ---------------------------------------

cursor.execute("SELECT COUNT(*) FROM matches")

total_matches = cursor.fetchone()[0]

print("\n========== DATABASE SUMMARY ==========")

print("Database:", DB_PATH)

print("Total matches stored:", total_matches)

cursor.execute("""
SELECT HomeTeam, AwayTeam, FTHG, FTAG, FTR
FROM matches
LIMIT 5
""")

print("\nSample database records:")

for row in cursor.fetchall():
    print(row)


# ---------------------------------------
# 8. CLOSE DATABASE
# ---------------------------------------

connection.commit()
connection.close()

print("\nFootIQ database created successfully!")
