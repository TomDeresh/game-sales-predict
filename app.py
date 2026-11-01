
import streamlit as st
import kagglehub
import pandas as pd
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(
    page_title="Beat the AI",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 Beat the AI!")
st.subheader("Can you predict game sales better than AI?")

st.write(
    "You get a real game, its user rating and release year. "
    "Your mission is to predict its worldwide sales!"
)

# -------------------------
# LOAD DATA AND TRAIN MODEL
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
        raise FileNotFoundError("No CSV file found!")

    df = pd.read_csv(csv_files[0])

    # Convert User_Score to numeric
    df["User_Score"] = pd.to_numeric(
        df["User_Score"],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna(
        subset=[
            "User_Score",
            "Year_of_Release",
            "Global_Sales"
        ]
    )

    # Keep valid ratings
    df = df[
        (df["User_Score"] >= 0) &
        (df["User_Score"] <= 10)
    ]

    # Features
    X = df[[
        "User_Score",
        "Year_of_Release"
    ]]

    # Target: worldwide sales in millions
    y = df["Global_Sales"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train AI
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Keep only test games for the competition
    test_data = df.loc[X_test.index].copy()

    return model, test_data, len(df)


# -------------------------
# LOAD MODEL
# -------------------------

try:

    model, test_data, total_games = load_data()

    st.success(
        f"Successfully loaded {total_games:,} real games!"
    )

except Exception as e:

    st.error(f"Error loading dataset: {e}")
    st.stop()


# -------------------------
# SESSION STATE
# -------------------------

if "game" not in st.session_state:

    st.session_state.game = test_data.sample(
        1
    ).iloc[0]

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

st.write("Here is your mystery game:")

st.metric(
    "⭐ User Rating",
    f"{game['User_Score']:.1f}/10"
)

st.metric(
    "📅 Release Year",
    int(game["Year_of_Release"])
)

st.info(
    "Predict worldwide sales in millions of copies."
)


# -------------------------
# PLAYER GUESS
# -------------------------

guess = st.number_input(
    "🧠 Your prediction (millions of copies)",
    min_value=0.0,
    max_value=1000.0,
    value=1.0,
    step=0.5
)


if st.button(
    "🚀 Submit Guess",
    use_container_width=True
):

    st.session_state.guess = guess
    st.session_state.submitted = True
    st.rerun()


# -------------------------
# RESULTS
# -------------------------

if st.session_state.submitted:

    actual = game["Global_Sales"]

    X_game = pd.DataFrame([{
        "User_Score": game["User_Score"],
        "Year_of_Release": game["Year_of_Release"]
    }])

    # AI prediction
    ai_prediction = model.predict(X_game)[0]

    # Avoid negative sales predictions
    ai_prediction = max(0, ai_prediction)

    # Calculate errors
    player_error = abs(
        st.session_state.guess - actual
    )

    ai_error = abs(
        ai_prediction - actual
    )

    st.divider()

    st.header("📊 Results")

    # Reveal game name
    st.write(f"🎮 **Game:** {game['Name']}")

    st.metric(
        "🧠 Your prediction",
        f"{st.session_state.guess:.2f} million"
    )

    st.metric(
        "🤖 AI prediction",
        f"{ai_prediction:.2f} million"
    )

    st.metric(
        "🏆 Actual worldwide sales",
        f"{actual:.2f} million"
    )

    st.divider()

    st.subheader("Prediction Errors")

    col1, col2 = st.columns(2)

    col1.metric(
        "Your error",
        f"{player_error:.2f} million"
    )

    col2.metric(
        "AI error",
        f"{ai_error:.2f} million"
    )

    # Winner
    if player_error < ai_error:

        st.success("🏆 YOU BEAT THE AI!")

        st.session_state.score += 1

    elif player_error > ai_error:

        st.error("🤖 AI WINS THIS ROUND!")

    else:

        st.info("🤝 IT'S A TIE!")

    st.session_state.rounds += 1

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
# NEXT ROUND
# -------------------------

st.divider()

if st.button(
    "🎮 Next Game",
    use_container_width=True
):

    st.session_state.game = test_data.sample(
        1
    ).iloc[0]

    st.session_state.submitted = False

    st.rerun()


# -------------------------
# RESET SCORE
# -------------------------

if st.button("🔄 Reset Score"):

    st.session_state.score = 0
    st.session_state.rounds = 0

    st.session_state.game = test_data.sample(
        1
    ).iloc[0]

    st.session_state.submitted = False

    st.rerun()


# -------------------------
# FOOTER
# -------------------------

st.divider()

st.caption(
    "Dataset: Kaggle - Video Game Sales with Ratings"
)

st.caption(
    "Machine Learning: Linear Regression"
)

st.caption(
    "Features: User Score + Release Year"
)

st.caption(
    "Target: Global Sales (millions of copies)"
)
