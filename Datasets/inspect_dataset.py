
import pandas as pd
from pathlib import Path

# Find the CSV file in the current folder
csv_files = list(Path(".").glob("*.csv"))

if not csv_files:
    print("No CSV file found! Place the CSV in this folder.")
    exit()

file_path = csv_files[0]
print("Dataset:", file_path.name)

# Load dataset
df = pd.read_csv(
    "C:/Users/hp/OneDrive/Desktop/FootIQ/Datasets/results.csv",
    encoding="cp1252"
)

# Basic information
print("\n--- DATASET OVERVIEW ---")
print("Total matches:", len(df))
print("Total columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

# Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sort_values(ascending=False))

# Check essential prediction columns
essential = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

print("\n--- ESSENTIAL COLUMNS ---")
for col in essential:
    if col in df.columns:
        print(col, "Missing:", df[col].isnull().sum())

# Identify date column
date_col = next(
    (c for c in ["DateTime", "Date"] if c in df.columns),
    None
)

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year

    print("\n--- MATCHES BY YEAR ---")
    print(df.groupby("Year").size())

    print("\n--- MISSING STATISTICS BY YEAR ---")
    stats = ["HS", "AS", "HST", "AST", "HC", "AC",
             "HY", "AY", "HR", "AR"]

    available = [c for c in stats if c in df.columns]

    print(df.groupby("Year")[available].apply(
        lambda x: x.isnull().sum()
    ).to_string())

print("\n--- FIRST FIVE MATCHES ---")
print(df.head().to_string())


print("\n--- MATCH RESULT DISTRIBUTION ---")
print(df["FTR"].value_counts())
print("\nResult percentages:")
print(df["FTR"].value_counts(normalize=True) * 100)

print("\n--- TEAM INFORMATION ---")
teams = pd.concat([df["HomeTeam"], df["AwayTeam"]])

print("Unique teams:", teams.nunique())
print("\nTeams:")
print(sorted(teams.unique()))

print("\n--- DATE RANGE ---")
print("First match:", df["DateTime"].min())
print("Last match:", df["DateTime"].max())

print("\n--- DUPLICATE MATCHES ---")
print("Duplicate rows:", df.duplicated().sum())

print("\n--- SEASON INFORMATION ---")
print("Total seasons:", df["Season"].nunique())
print(sorted(df["Season"].unique()))
