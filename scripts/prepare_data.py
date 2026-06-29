import pandas as pd

INPUT = "data/tracks.csv"
OUTPUT = "data/tracks_small.csv"
SAMPLE_SIZE = 5000

df = pd.read_csv(INPUT)

print(f"Full dataset: {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# drop rows where essential fields are missing
df = df.dropna(subset=["name", "artists"])

# sample randomly since there's no genre column
df_small = (
    df.sample(SAMPLE_SIZE, random_state=42)
    .reset_index(drop=True)
)

# keep only the columns we actually need
df_small = df_small[[
    "name", "artists", "album",
    "danceability", "energy", "valence",
    "tempo", "acousticness", "instrumentalness",
    "year"
]]

# rename for clarity
df_small = df_small.rename(columns={"name": "track_name"})

df_small.to_csv(OUTPUT, index=False)
print(f"\nSaved {len(df_small)} rows to {OUTPUT}")
print(f"\nFirst 3 rows:")
print(df_small.head(3).to_string())