
import pandas as pd

# 1. Load the original dataset
df = pd.read_csv("results.csv", encoding="cp1252")

print("Original dataset shape:", df.shape)

# 2. Convert match date into datetime format
df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    errors="coerce",
    utc=True
)

# 3. Remove rows only if essential information is missing
essential_columns = [
    "DateTime",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR"
]

df = df.dropna(subset=essential_columns)

# 4. Convert match outcome into numerical values
result_mapping = {
    "H": 0,
    "D": 1,
    "A": 2
}

df["Target"] = df["FTR"].map(result_mapping)

# 5. Sort matches chronologically
df = df.sort_values("DateTime").reset_index(drop=True)

# 6. Display the cleaned dataset information
print("\n--- CLEANED DATASET ---")
print("Total matches:", len(df))
print("Total columns:", len(df.columns))

print("\nTarget distribution:")
print(df["Target"].value_counts().sort_index())

print("\nFirst five cleaned matches:")
print(df[[
    "DateTime",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "Target"
]].head())

# 7. Save the cleaned dataset
df.to_csv("matches_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
