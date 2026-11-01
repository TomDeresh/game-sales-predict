
import streamlit as st
import kagglehub
import pandas as pd
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Beat the AI",
    page_icon="🎮",
    layout="centered"
)

# -------------------------
# Load dataset and train AI
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
        raise FileNotFoundError("No CSV file found in dataset")

    df = pd.read_csv(csv_files[0])

    # Keep only rows with all required values
    df = df.dropna(
        subset=[
            "Critic_Score",
            "Year_of_Release",
            "Global_Sales"
        ]
    )

    # Features and target
    X = df[["Critic_Score", "Year_of_Release"]]
    y = df["Global_Sales"]

    # Separate training data from test data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test dataset
    test_data = df.loc[X_test.index].copy()

    return model, test_data, len(df)


# -------------------------
# Load everything
# -------------------------

st.title("🎮 Beat the AI!")
st.subheader("Can you predict game sales better than AI?")

try:
    model, test_data, total_games = load_data()

except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.success(f"Loaded {total_games:,} real games from Kaggle!")

# -------------------------
# Session state
# -------------------------

if "game" not in st.session_state:
    st.session_state.game = test_data.sample(1).iloc[0]

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "score" not in st.session_state:
    st.session_state.score = 0

if "rounds" not in st.session_state:
    st.session_state.rounds = 0


game = st.session_state.game

# -------------------------
# Game information
# -------------------------

st.divider()

st.subheader("🎯 Your challenge")

st.write("Predict the worldwide sales of this game!")

st.metric(
    "⭐ Critics Rating",
    f"{game['Critic_Score']}/100"
)

st.metric(
    "📅 Release Year",
    int(game["Year_of_Release"])
)

guess = st.number_input(
    "How many million copies do you think it sold?",
    min_value=0.0,
    value=1.0,
    step=0.5
)

# -------------------------
# Submit guess
# -------------------------

if st.button("🚀 Submit Guess", use_container_width=True):

    st.session_state.guess = guess
    st.session_state.submitted = True
    st.rerun()

# -------------------------
# Results
# -------------------------

if st.session_state.submitted:

    actual = game["Global_Sales"]

    X_game = pd.DataFrame([{
        "Critic_Score": game["Critic_Score"],
        "Year_of_Release": game["Year_of_Release"]
    }])

    ai_prediction = model.predict(X_game)[0]

    player_error = abs(st.session_state.guess - actual)
    ai_error = abs(ai_prediction - actual)

    st.divider()
    st.subheader("📊 The Results!")

    st.write(f"🎮 Game: **{game['Name']}**")

    st.metric(
        "Your prediction",
        f"{st.session_state.guess:.2f} million"
    )

    st.metric(
        "🤖 AI prediction",
        f"{max(0, ai_prediction):.2f} million"
    )

    st.metric(
        "🏆 Actual sales",
        f"{actual:.2f} million"
    )

    st.write("### Prediction errors")

    st.write(f"Your error: {player_error:.2f} million")
    st.write(f"AI error: {ai_error:.2f} million")

    if player_error < ai_error:
        st.success("🏆 YOU BEAT THE AI!")
        st.session_state.score += 1

    elif player_error > ai_error:
        st.error("🤖 AI WON THIS ROUND!")

    else:
        st.info("🤝 It's a tie!")

    st.session_state.rounds += 1

    st.divider()

    st.subheader("🏅 Scoreboard")

    st.write(f"Your wins: {st.session_state.score}")
    st.write(f"Rounds played: {st.session_state.rounds}")

# -------------------------
# Next round
# -------------------------

if st.button("🎮 Next Game", use_container_width=True):

    st.session_state.game = test_data.sample(1).iloc[0]
    st.session_state.submitted = False

    st.rerun()

if st.button("🔄 Reset Score"):

    st.session_state.score = 0
    st.session_state.rounds = 0
    st.session_state.game = test_data.sample(1).iloc[0]
    st.session_state.submitted = False

    st.rerun()

st.divider()

st.caption(
    "Dataset: Kaggle - Video Game Sales with Ratings"
)
st.caption(
    "AI Model: Linear Regression | Features: Critic Score + Release Year"
)
