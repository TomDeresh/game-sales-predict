
import streamlit as st
import kagglehub
import pandas as pd
import os
import glob

# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(
    page_title="Beat the AI",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 Beat the AI!")
st.subheader("Can you beat the original AI model?")

st.write(
    "Predict the worldwide sales of real video games "
    "better than the AI!"
)

# -------------------------
# ORIGINAL MODEL WEIGHTS
# -------------------------

w1 = 110212.26226094973
w2 = 7643.05127624
b = -45899.861928343424

# -------------------------
# LOAD KAGGLE DATASET
# -------------------------

@st.cache_data
def load_data():

    path = kagglehub.dataset_download(
        "rush4ratio/video-game-sales-with-ratings"
    )

    csv_files = glob.glob(
        os.path.join(path, "**", "*.csv"),
        recursive=True
    )

    if not csv_files:
        raise FileNotFoundError("CSV file not found")

    df = pd.read_csv(csv_files[0])

    # Convert user ratings to numbers
    df["User_Score"] = pd.to_numeric(
        df["User_Score"],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna(
        subset=[
            "Name",
            "User_Score",
            "Year_of_Release",
            "Global_Sales"
        ]
    )

    # Validate ratings
    df = df[
        (df["User_Score"] >= 0) &
        (df["User_Score"] <= 10)
    ]

    return df


# -------------------------
# LOAD DATA
# -------------------------

try:

    df = load_data()

    st.success(
        f"Loaded {len(df):,} real games from Kaggle!"
    )

except Exception as e:

    st.error(f"Dataset loading error: {e}")
    st.stop()


# -------------------------
# SESSION STATE
# -------------------------

if "game" not in st.session_state:
    st.session_state.game = df.sample(1).iloc[0]

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "score" not in st.session_state:
    st.session_state.score = 0

if "rounds" not in st.session_state:
    st.session_state.rounds = 0


game = st.session_state.game


# -------------------------
# GAME INFORMATION
# -------------------------

st.divider()

st.header("🎯 Your Challenge")

st.write("Guess the worldwide sales of this mystery game!")

st.metric(
    "⭐ User Rating",
    f"{game['User_Score']:.1f}/10"
)

st.metric(
    "📅 Release Year",
    int(game["Year_of_Release"])
)

st.info(
    "Sales are measured in millions of copies."
)


# -------------------------
# PLAYER GUESS
# -------------------------

guess = st.number_input(
    "🧠 Your sales prediction",
    min_value=0.0,
    max_value=1000.0,
    value=1.0,
    step=0.5
)


# -------------------------
# SUBMIT GUESS
# -------------------------

if st.button(
    "🚀 Submit Guess",
    use_container_width=True
):

    st.session_state.guess = guess
    st.session_state.submitted = True
    st.rerun()


# -------------------------
# CALCULATE RESULTS
# -------------------------

if st.session_state.submitted:

    actual = float(game["Global_Sales"])

    # Convert user rating from 0-10 to 0-100
    rating = float(game["User_Score"]) * 10

    year = float(game["Year_of_Release"])

    # Original AI formula
    ai_prediction = (
        w1 * rating +
        w2 * year +
        b
    )

    # Avoid negative predictions
    ai_prediction = max(0, ai_prediction)

    # Calculate errors
    player_error = abs(
        st.session_state.guess - actual
    )

    ai_error = abs(
        ai_prediction - actual
    )

    # -------------------------
    # RESULTS
    # -------------------------

    st.divider()

    st.header("📊 Results")

    st.write(f"🎮 **Game:** {game['Name']}")

    st.metric(
        "🧠 Your Prediction",
        f"{st.session_state.guess:.2f} million"
    )

    st.metric(
        "🤖 AI Prediction",
        f"{ai_prediction:.2f} million"
    )

    st.metric(
        "🏆 Actual Sales",
        f"{actual:.2f} million"
    )

    st.divider()

    st.subheader("Prediction Errors")

    col1, col2 = st.columns(2)

    col1.metric(
        "Your Error",
        f"{player_error:.2f} million"
    )

    col2.metric(
        "AI Error",
        f"{ai_error:.2f} million"
    )

    # -------------------------
    # WINNER
    # -------------------------

    if player_error < ai_error:

        st.success("🏆 YOU BEAT THE AI!")

        st.session_state.score += 1

    elif player_error > ai_error:

        st.error("🤖 AI WINS THIS ROUND!")

    else:

        st.info("🤝 IT'S A TIE!")

    # Update rounds
    st.session_state.rounds += 1

    # -------------------------
    # SCOREBOARD
    # -------------------------

    st.divider()

    st.header("🏅 Scoreboard")

    col1, col2 = st.columns(2)

    col1.metric(
        "Your Wins",
        st.session_state.score
    )

    col2.metric(
        "Rounds Played",
        st.session_state.rounds
    )


# -------------------------
# NEXT GAME
# -------------------------

st.divider()

if st.button(
    "🎮 Next Game",
    use_container_width=True
):

    st.session_state.game = df.sample(1).iloc[0]

    st.session_state.submitted = False

    st.rerun()


# -------------------------
# RESET SCORE
# -------------------------

if st.button("🔄 Reset Score"):

    st.session_state.score = 0
    st.session_state.rounds = 0

    st.session_state.game = df.sample(1).iloc[0]

    st.session_state.submitted = False

    st.rerun()


# -------------------------
# FOOTER
# -------------------------

st.divider()

st.caption("Dataset: Kaggle - Video Game Sales with Ratings")

st.caption("Model: Original Linear Regression Weights")

st.caption("Features: User Rating + Release Year")

st.caption("Target: Global Sales (Millions of Copies)")
