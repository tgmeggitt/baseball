import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import batting_stats
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1) Constants for wRAA and Runs calculations — tune these as needed
LEAGUE_WOBA = 0.320      # Typical league average wOBA
WOBA_SCALE = 1.25        # Typical scaling factor
LEAGUE_RUNS_PER_PA = 0.115  # Approx. runs/PA in recent MLB seasons

# Set Streamlit page configuration
st.set_page_config(page_title="MLB wOBA Predictor", page_icon="⚾", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
        body { background-color: #F5F5F5; }
        .stApp { background-color: #F5F5F5; }
        .stTextInput input { font-size: 18px; }
        .stMarkdown h1 { color: #0033A0; text-align: center; }
        .stMarkdown h2 { color: #C8102E; }
        .stMarkdown p { color: #333333; }
        .stButton>button { background-color: #C8102E; color: white; border-radius: 10px; }
        .stButton>button:hover { background-color: #0033A0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define paths for model and scaler
MODEL_PATH = "C:/Users/tgmeg/Desktop/wOBA Prediction Model/Web App/wOBA_NN_model.h5"
SCALER_PATH = "C:/Users/tgmeg/Desktop/wOBA Prediction Model/Web App/scaler.pkl"

# Load the trained model
custom_objects = {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}
model = load_model(MODEL_PATH, custom_objects=custom_objects)

# Load the scaler
scaler = joblib.load(SCALER_PATH)

# Display MLB Logo
st.image("C:/Users/tgmeg/Desktop/wOBA Prediction Model/Web App/logo.png")

# Define feature order (must match training features)
FEATURES = [
    "WPA/LI", "REW", "RE24", "xSLG", "Off", "EV", "wRAA", "ISO+",
    "max_exit_velocity", "BB_K_Ratio", "WPA", "Speed_Impact",
    "rel_wOBA", "rel_ISO", "rel_EV", "rel_OBP", "rel_SLG", "rel_hard_hit_rate", "Age",
    "wOBA_change", "age_decline_factor", "weighted_wOBA_3Y", "weighted_wOBA_5Y"
]

# Fetch player names (for auto-complete)
years = list(range(2015, 2024))
df = pd.concat([batting_stats(year) for year in years], ignore_index=True)
df["Name"] = df["Name"].str.title()  # Ensure consistent casing
player_names = sorted(df["Name"].unique().tolist())


def fetch_and_prepare_player_data(player_name, start_year=2015, end_year=2023):
    """
    Fetch player stats and compute necessary features. Returns the 'latest_data'
    for the given player, plus the corresponding 'PA' (plate appearances) which we now hardcode to 600.
    """
    all_years_data = pd.concat(
        [batting_stats(year) for year in range(start_year, end_year + 1)],
        ignore_index=True
    )

    # Rename columns for consistency
    all_years_data.rename(columns={
        "IDfg": "player_id",
        "Season": "year",
        "HardHit%": "hard_hit_rate",
        "maxEV": "max_exit_velocity",
        "BB%": "bb_rate",
        "K%": "k_rate",
        "Spd": "speed_score"
    }, inplace=True)

    all_years_data["Name"] = all_years_data["Name"].str.title()

    # Filter for the player
    player_data = all_years_data[all_years_data["Name"] == player_name].copy()

    if player_data.empty:
        st.warning(f"No data found for {player_name} in years {start_year}-{end_year}")
        return None, 0

    player_data = player_data.sort_values(by=["player_id", "year"])

    # Compute wOBA Change
    player_data["wOBA_change"] = player_data.groupby("player_id")["wOBA"].diff()

    # Compute league averages
    league_avg = all_years_data.groupby("year")[["wOBA", "ISO", "EV", "OBP", "SLG", "hard_hit_rate"]].mean()
    for stat in ["wOBA", "ISO", "EV", "OBP", "SLG", "hard_hit_rate"]:
        player_data[f"rel_{stat}"] = player_data[stat] - player_data["year"].map(league_avg[stat])

    # Compute weighted rolling averages
    def compute_weighted_rolling(df, col, weights):
        return df[col].rolling(len(weights), min_periods=len(weights)).apply(
            lambda x: np.dot(x, weights[-len(x):]), raw=True
        )

    player_data["weighted_wOBA_3Y"] = compute_weighted_rolling(player_data, "wOBA", [0.5, 0.3, 0.2])
    player_data["weighted_wOBA_5Y"] = compute_weighted_rolling(player_data, "wOBA", [0.4, 0.25, 0.15, 0.1, 0.1])

    # Compute BB/K Ratio
    player_data["BB_K_Ratio"] = player_data["bb_rate"] / player_data["k_rate"]

    # Compute Speed Impact
    player_data["Speed_Impact"] = player_data["speed_score"] * player_data["ISO+"]

    # Compute Age-Based Decline Factor
    def age_decline_factor(age):
        return 1 / (1 + np.exp((age - 30) / 3))

    player_data["age_decline_factor"] = player_data["Age"].apply(age_decline_factor)

    # Ensure all required columns exist
    for col in FEATURES:
        if col not in player_data.columns:
            player_data[col] = 0

    # Select the latest season's data
    latest_data = player_data[player_data["year"] == end_year]

    if latest_data.empty:
        # If no row for exactly end_year, fallback to the last row for that player
        latest_data = player_data.iloc[-1:]

    # Hardcode PA to 600, ignoring whatever is in data
    pa_value = 600

    # Keep only the needed features
    latest_data = latest_data.fillna(0)
    latest_data = latest_data[FEATURES]

    return latest_data, pa_value


st.title("MLB wOBA Prediction App ⚾")
st.write("Select 9 players to predict wOBA, then compute wRAA and Runs for each.")

player_names_selected = []
for i in range(9):
    player_name = st.selectbox(
        f"Player {i+1}",
        player_names,
        key=f"player_{i}",
        index=None,
        placeholder="Player Name"
    )
    if player_name:
        player_names_selected.append(player_name)

if st.button("Predict wOBA, wRAA, and Runs"):
    results = []
    total_runs = 0.0

    for player_name in player_names_selected:
        # Fetch data & PA
        player_features, player_PA = fetch_and_prepare_player_data(player_name)
        if player_features is None:
            continue  # skip if no data

        # Align features
        missing_features = [col for col in FEATURES if col not in player_features.columns]
        extra_features = [col for col in player_features.columns if col not in FEATURES]

        # Fill missing features with 0 if needed
        for col in missing_features:
            player_features[col] = 0

        if extra_features:
            player_features = player_features[FEATURES]

        # Scale, then predict wOBA
        player_features_scaled = scaler.transform(player_features)
        predicted_wOBA = model.predict(player_features_scaled)[0][0]

        # Compute wRAA = ((wOBA - league_wOBA) / wOBA_SCALE) * PA
        wRAA = ((predicted_wOBA - LEAGUE_WOBA) / WOBA_SCALE) * player_PA

        # Compute Runs = wRAA + (league_runs_per_PA * PA)
        runs = wRAA + (LEAGUE_RUNS_PER_PA * player_PA)
        total_runs += runs

        results.append({
            "Player": player_name,
            "Predicted wOBA": round(predicted_wOBA, 3),
            "PA": player_PA,
            "wRAA": round(wRAA, 2),
            "Runs": round(runs, 2)
        })

    # Display results
    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        st.subheader(f"Total Runs (9 Players): {round(total_runs, 2)}")
