# Moodify 🎵

A semantic music search engine that finds songs based on how you're feeling.
Describe your mood in plain English and Moodify returns the most relevant tracks
from a dataset of 5,000 Spotify songs — powered by vector embeddings and Qdrant.

## How it works

1. Each song is converted into a natural language description using its audio features
   (energy, valence, danceability, tempo, acousticness)
2. These descriptions are embedded into 384-dimensional vectors using
   `sentence-transformers` (`all-MiniLM-L6-v2`)
3. Vectors are stored in a Qdrant cloud collection
4. At search time, your mood query is embedded the same way and the
   closest matching song vectors are returned using cosine similarity

## Tech stack

| Layer | Tool |
|---|---|
| Vector database | Qdrant (cloud, free tier) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Dataset | Spotify 1.2M+ Songs (Kaggle) |
| UI | Streamlit |
| Language | Python 3.12 |

## Project structure

```
moodify/
├── app.py                  # Streamlit UI
├── scripts/
│   ├── prepare_data.py     # trims dataset to 5,000 songs
│   ├── ingest.py           # generates embeddings and uploads to Qdrant
│   └── search.py           # CLI mood search
├── data/                   # dataset lives here (not committed)
├── .streamlit/
│   └── config.toml         # disables noisy file watcher
├── .env.example            # env variable template
└── requirements.txt
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/moodify.git
cd moodify
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Copy `.env.example` to `.env` and fill in your Qdrant credentials:
```
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

Get these from [cloud.qdrant.io](https://cloud.qdrant.io) — free account, no card needed.

**5. Download the dataset**

Download `Spotify 1.2M+ Songs` from Kaggle and place `tracks.csv` inside the `data/` folder.

**6. Prepare and index the data**
```bash
python scripts/prepare_data.py   # trims to 5,000 songs
python scripts/ingest.py         # uploads vectors to Qdrant
```

**7. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Example queries

- `rainy sad evening`
- `metal gym session`
- `chill sunday morning`
- `happy summer road trip`
- `late night studying`

## What I learned

- How vector databases store and search embeddings using cosine similarity
- How to convert structured data into natural language for embedding
- How semantic search differs from keyword search
- Qdrant collections, upserts, and querying via the Python client
- Managing secrets safely with `.env` and `.gitignore`