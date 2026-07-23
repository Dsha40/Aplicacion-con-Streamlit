"""
Music Genre Classifier — Streamlit app
Predicts the musical genre of a track from its Spotify audio features,
using a Random Forest model trained on real Spotify data (10 genres).
"""

import os
from pickle import load

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Genre Classifier",
    page_icon="🎧",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load model (once, cached across reruns)
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "genre_classifier_real.sav")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = load(f)
    return bundle["model"], bundle["scaler"], bundle["feature_cols"]


model, scaler, feature_cols = load_model()

GENRE_EMOJI = {
    "blues": "🎷", "classical": "🎻", "country": "🤠", "edm": "🎛️",
    "heavy-metal": "🤘", "hip-hop": "🎤", "jazz": "🎺", "pop": "⭐",
    "reggae": "🌴", "rock": "🎸",
}

# ---------------------------------------------------------------------------
# Custom styling (Spotify-inspired dark theme)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp { background-color: #0d0d0d; }
    h1, h2, h3 { color: #1DB954 !important; }
    .genre-result {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #1DB954 0%, #14833b 100%);
        margin-top: 1rem;
    }
    .genre-result h2 { color: white !important; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }
    .genre-result p { color: #d9f2e3; margin: 0.3rem 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎧 Music Genre Classifier")
st.caption("Trained on real Spotify audio features · Random Forest · 10 genres")

st.write(
    "Adjust the audio characteristics of a track below, and the model will "
    "predict the most likely genre — trained on danceability, energy, "
    "acousticness and other real Spotify audio features."
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form("track_form"):
    st.subheader("Audio Features")

    col1, col2 = st.columns(2)
    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.6, 0.01)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.05, 0.01)
        valence = st.slider("Valence (positivity)", 0.0, 1.0, 0.5, 0.01)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.08, 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)

    with col2:
        loudness = st.slider("Loudness (dB)", -35.0, 0.0, -8.0, 0.5)
        tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 118.0, 1.0)
        popularity = st.slider("Popularity", 0, 100, 45, 1)
        duration_min = st.slider("Duration (minutes)", 1.0, 10.0, 3.5, 0.1)
        key = st.selectbox("Key", list(range(12)), index=0)
        mode = st.selectbox("Mode", ["Minor", "Major"], index=1)
        time_signature = st.selectbox("Time Signature", [3, 4, 5], index=1)
        explicit = st.checkbox("Explicit content", value=False)

    submitted = st.form_submit_button("🎵 Predict Genre", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if submitted:
    track = pd.DataFrame([{
        "danceability": danceability,
        "energy": energy,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "valence": valence,
        "loudness": loudness,
        "tempo": tempo,
        "speechiness": speechiness,
        "liveness": liveness,
        "popularity": popularity,
        "duration_ms": duration_min * 60_000,
        "key": key,
        "mode": 1 if mode == "Major" else 0,
        "time_signature": time_signature,
        "explicit": int(explicit),
    }])[feature_cols]

    # Envolvemos de vuelta en un DataFrame con los mismos nombres de columna que vio
    # el modelo al entrenar -- evita un warning de sklearn (y es mas robusto a futuro)
    track_scaled = pd.DataFrame(scaler.transform(track), columns=feature_cols)
    proba = model.predict_proba(track_scaled)[0]
    predicted_genre = model.classes_[np.argmax(proba)]
    confidence = proba.max() * 100

    emoji = GENRE_EMOJI.get(predicted_genre, "🎵")
    st.markdown(
        f"""
        <div class="genre-result">
            <h2>{emoji} {predicted_genre}</h2>
            <p>Confidence: {confidence:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Full probability breakdown")
    proba_df = pd.DataFrame({
        "Genre": model.classes_,
        "Probability": proba,
    }).sort_values("Probability", ascending=False).reset_index(drop=True)
    proba_df["Probability"] = proba_df["Probability"].apply(lambda x: f"{x*100:.1f}%")

    st.dataframe(proba_df, use_container_width=True, hide_index=True)

    st.bar_chart(
        pd.DataFrame({"Genre": model.classes_, "Probability": proba}).set_index("Genre"),
        color="#1DB954",
    )

st.divider()
st.caption(
    "Model: Random Forest, trained on a real Spotify tracks dataset "
    "(maharshipandya/spotify-tracks-dataset). For educational purposes."
)
