import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "moodify"

st.set_page_config(page_title="Moodify", page_icon="🎵", layout="centered")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

model = load_model()
client = load_client()

# header
st.title("🎵 Moodify")
st.markdown("Describe how you're feeling and find songs that match your mood.")
st.divider()

# search bar
mood = st.text_input(
    label="What's your mood?",
    placeholder="e.g. rainy sad evening, energetic gym session, chill sunday morning...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([3, 1])
with col1:
    num_results = st.slider("Number of results", min_value=3, max_value=15, value=5)
with col2:
    search_clicked = st.button("Find songs ↗", use_container_width=True)

st.divider()

# search logic
if search_clicked and mood.strip():
    with st.spinner("searching..."):
        query_vector = model.encode(mood).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=num_results,
        ).points

    st.markdown(f"**Top {len(results)} songs for:** *{mood}*")
    st.write("")

    for i, hit in enumerate(results, 1):
        p = hit.payload

        with st.container():
            col_num, col_info, col_score = st.columns([0.5, 6, 1.5])

            with col_num:
                st.markdown(f"### {i}")

            with col_info:
                st.markdown(f"**{p['track_name']}**")
                st.caption(f"{p['artists']} · {p['album']} · {p['year']}")

                # mood bars
                bar_col1, bar_col2, bar_col3 = st.columns(3)
                with bar_col1:
                    st.metric("energy", round(p['energy'], 2))
                with bar_col2:
                    st.metric("happiness", round(p['valence'], 2))
                with bar_col3:
                    st.metric("danceability", round(p['danceability'], 2))

            with col_score:
                st.metric("match", f"{round(hit.score * 100, 1)}%")

            st.divider()

elif search_clicked and not mood.strip():
    st.warning("type a mood first!")

else:
    st.markdown(
        "<p style='color: var(--text-muted); text-align: center; padding-top: 2rem;'>"
        "type a mood above and hit find songs</p>",
        unsafe_allow_html=True
    )