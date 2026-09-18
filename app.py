import streamlit as st

st.title("tomProject")

user_prompt = st.number_input("Enter a game rating")
w = 110212.26226094973
b = -45899.861928343424
if user_prompt:
    response = w * user_prompt + b

    st.subheader("Response:")
    st.write(response)
