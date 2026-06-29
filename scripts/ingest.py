import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "moodify"
BATCH_SIZE = 100

def describe_song(row):
    """Turn a song's audio features into a natural language description."""
    energy = "high energy" if row["energy"] > 0.6 else "low energy" if row["energy"] < 0.4 else "moderate energy"
    mood = "happy and upbeat" if row["valence"] > 0.6 else "sad and melancholic" if row["valence"] < 0.4 else "neutral mood"
    dance = "very danceable" if row["danceability"] > 0.6 else "not very danceable"
    acoustic = "acoustic" if row["acousticness"] > 0.5 else "electronic"
    instrumental = "instrumental" if row["instrumentalness"] > 0.5 else "has vocals"
    tempo = "fast tempo" if row["tempo"] > 120 else "slow tempo"

    return (
        f"{row['track_name']} by {row['artists']}. "
        f"This song is {mood}, {energy}, {dance}, {acoustic}, {instrumental}, {tempo}. "
        f"Released in {int(row['year'])} on the album {row['album']}."
    )

def main():
    print("loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("connecting to qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # create collection (delete first if it already exists)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"deleted existing collection '{COLLECTION_NAME}'")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"created collection '{COLLECTION_NAME}'")

    print("loading dataset...")
    df = pd.read_csv("data/tracks_small.csv")
    print(f"{len(df)} songs loaded")

    print("generating descriptions and embeddings...")
    descriptions = [describe_song(row) for _, row in df.iterrows()]

    # show a sample so we can see what's being embedded
    print(f"\nsample description:\n{descriptions[0]}\n")

    embeddings = model.encode(descriptions, show_progress_bar=True)

    print("\nuploading to qdrant in batches...")
    points = []
    for i, (_, row) in enumerate(df.iterrows()):
        points.append(PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "track_name": row["track_name"],
                "artists": row["artists"],
                "album": row["album"],
                "year": int(row["year"]),
                "energy": round(float(row["energy"]), 3),
                "valence": round(float(row["valence"]), 3),
                "danceability": round(float(row["danceability"]), 3),
                "tempo": round(float(row["tempo"]), 1),
            }
        ))

    # upload in batches
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="uploading batches"):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"\ndone! {len(points)} songs indexed in qdrant.")

if __name__ == "__main__":
    main()