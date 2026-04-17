import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="MLP Delay Predictor", page_icon="🤖", layout="centered")

st.title("🤖 MLP Delay Probability Demo")
st.markdown("""
Enter flight details below. The trained MLP neural network will estimate the probability
that this flight departs **15 or more minutes late**.
""")

# ── Load artifacts (cached so they only load once) ───────────────────────────
@st.cache_resource
def load_artifacts():
    model       = tf.keras.models.load_model("model.keras")
    scaler      = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    airports    = joblib.load("top50_airports.pkl")
    return model, scaler, feature_cols, airports

try:
    model, scaler, feature_cols, airports = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

# ── Carrier metadata ──────────────────────────────────────────────────────────
CARRIER_NAMES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air (AA regional)",
    "OH": "PSA Airlines (AA regional)",
    "OO": "SkyWest Airlines",
    "YX": "Republic Airways",
}

MONTH_NAMES = {
    1:"January", 2:"February", 3:"March", 4:"April",
    5:"May", 6:"June", 7:"July", 8:"August",
    9:"September", 10:"October", 11:"November", 12:"December"
}

DAY_NAMES = {
    1:"Monday", 2:"Tuesday", 3:"Wednesday", 4:"Thursday",
    5:"Friday", 6:"Saturday", 7:"Sunday"
}

# ── Input widgets ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    carrier_label = st.selectbox(
        "Carrier",
        options=list(CARRIER_NAMES.keys()),
        format_func=lambda x: f"{x} – {CARRIER_NAMES[x]}"
    )
    month = st.selectbox(
        "Month",
        options=list(MONTH_NAMES.keys()),
        format_func=lambda x: MONTH_NAMES[x],
        index=5  # default June
    )
    day_of_week = st.selectbox(
        "Day of Week",
        options=list(DAY_NAMES.keys()),
        format_func=lambda x: DAY_NAMES[x]
    )

with col2:
    dep_hour = st.slider("Scheduled Departure Hour", min_value=0, max_value=23, value=8)
    day_of_month = st.slider("Day of Month", min_value=1, max_value=31, value=15)
    distance = st.number_input("Flight Distance (miles)", min_value=50, max_value=5000, value=500, step=50)

# Origin / Dest — show top-50 + Other
origin_options = sorted(airports["origin"]) + ["Other"]
dest_options   = sorted(airports["dest"])   + ["Other"]

col3, col4 = st.columns(2)
with col3:
    origin = st.selectbox("Origin Airport", origin_options)
with col4:
    dest   = st.selectbox("Destination Airport", dest_options)

# ── Inference ─────────────────────────────────────────────────────────────────
def build_input_row(carrier, month, dow, dep_hour, dom, distance, origin, dest,
                    feature_cols, scaler, airports):
    """Build a single preprocessed row matching the training feature space."""

    row = pd.DataFrame([{
        "OP_UNIQUE_CARRIER": carrier,
        "MONTH":             month,
        "DAY_OF_WEEK":       dow,
        "DEP_HOUR":          dep_hour,
        "DAY_OF_MONTH":      dom,
        "DISTANCE":          float(distance),
        "ORIGIN":            origin if origin in airports["origin"] else "Other",
        "DEST":              dest   if dest   in airports["dest"]   else "Other",
    }])

    # One-hot encode categoricals (same columns as training)
    cat_cols = ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "MONTH", "DAY_OF_WEEK"]
    row = pd.get_dummies(row, columns=cat_cols, drop_first=True, dtype=int)

    # Align to training feature space
    row = row.reindex(columns=feature_cols, fill_value=0)

    # Scale numerics
    num_cols = ["DAY_OF_MONTH", "DEP_HOUR", "DISTANCE"]
    row[num_cols] = scaler.transform(row[num_cols])

    return row.values.astype(np.float32)

if st.button("Predict Delay Probability", type="primary"):
    X_input = build_input_row(
        carrier_label, month, day_of_week, dep_hour, day_of_month,
        distance, origin, dest, feature_cols, scaler, airports
    )
    prob = float(model.predict(X_input, verbose=0)[0][0])

    st.divider()

    # Colour-coded result
    if prob < 0.30:
        colour = "green"
        verdict = "Low delay risk"
    elif prob < 0.50:
        colour = "orange"
        verdict = "Moderate delay risk"
    else:
        colour = "red"
        verdict = "High delay risk"

    st.markdown(
        f"### Estimated delay probability: "
        f"<span style='color:{colour}; font-size:2rem'><b>{prob:.1%}</b></span> "
        f"— {verdict}",
        unsafe_allow_html=True
    )

    # Progress bar as a simple gauge
    st.progress(prob)

    st.caption(
        f"The MLP assigns a **{prob:.1%} probability** that this "
        f"{CARRIER_NAMES.get(carrier_label, carrier_label)} flight from {origin} to {dest} "
        f"on {DAY_NAMES[day_of_week]}, {MONTH_NAMES[month]} at {dep_hour:02d}:00 "
        f"will depart 15+ minutes late."
    )

    st.markdown("""
    **How to interpret this:** The model was trained on ~480k flights with a ROC-AUC of 0.715.
    It learned patterns from carrier, time of day, seasonality, route, and distance, but
    does **not** have access to real-time weather or air traffic data, which limits its
    ceiling. Probabilities above 40% represent meaningfully elevated risk.
    """)
