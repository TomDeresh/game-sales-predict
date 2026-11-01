
import streamlit as st
import random

st.set_page_config(page_title="Beat the AI", page_icon="🎮")

# Model parameters
w1 = 110212.26226094973
w2 = 7643.05127624
b = -45899.861928343424

st.title("🎮 Beat the AI!")
st.subheader("Can you predict game revenue better than AI?")

st.write("Enter a game's rating and release year, then guess its revenue!")

# Session state
if "score" not in st.session_state:
    st.session_state.score = 0

if "round" not in st.session_state:
    st.session_state.round = 0

rating = st.number_input(
    "⭐ Game rating",
    min_value=0.0,
    max_value=10.0,
    value=8.0,
    step=0.1
)

year = st.number_input(
    "📅 Release year",
    min_value=1970,
    max_value=2026,
    value=2020
)

# AI prediction
ai_prediction = w1 * rating + w2 * year + b

st.divider()

st.subheader("🤖 AI prediction")
st.write(f"${ai_prediction:,.2f}")

st.divider()

st.subheader("🧠 Your turn!")

guess = st.number_input(
    "How much do you think the game earned ($)?",
    min_value=0.0,
    value=1000000.0,
    step=100000.0
)

# Actual revenue supplied by the game dataset
actual_revenue = st.number_input(
    "Actual game revenue ($) — reveal after guessing",
    min_value=0.0,
    value=1500000.0,
    step=100000.0
)

if st.button("🎯 Submit Guess", use_container_width=True):

    player_error = abs(guess - actual_revenue)
    ai_error = abs(ai_prediction - actual_revenue)

    st.session_state.round += 1

    st.subheader("📊 Results")

    st.metric("Your prediction", f"${guess:,.0f}")
    st.metric("AI prediction", f"${ai_prediction:,.0f}")
    st.metric("Actual revenue", f"${actual_revenue:,.0f}")

    st.write(f"Your error: ${player_error:,.0f}")
    st.write(f"AI error: ${ai_error:,.0f}")

    if player_error < ai_error:
        st.success("🏆 You BEAT THE AI!")
        st.session_state.score += 1

    elif player_error > ai_error:
        st.error("🤖 AI wins this round!")

    else:
        st.info("🤝 It's a tie!")

    st.write(f"### Your score: {st.session_state.score}")

if st.button("🔄 Reset Score"):
    st.session_state.score = 0
    st.session_state.round = 0
    st.rerun()
