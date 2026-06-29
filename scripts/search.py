import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "moodify"

def search(mood: str, limit: int = 5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    query_vector = model.encode(mood).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    ).points

    print(f"\ntop {limit} songs for mood: '{mood}'\n")
    print(f"{'#':<4} {'song':<45} {'artist':<30} {'score':<8} {'vibe'}")
    print("-" * 110)

    for i, hit in enumerate(results, 1):
        p = hit.payload
        vibe = f"energy={p['energy']} valence={p['valence']} dance={p['danceability']}"
        print(f"{i:<4} {p['track_name'][:43]:<45} {str(p['artists'])[:28]:<30} {round(hit.score, 4):<8} {vibe}")

if __name__ == "__main__":
    import sys
    mood = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "happy upbeat summer"
    search(mood)