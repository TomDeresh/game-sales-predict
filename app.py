import streamlit as st

st.title("tomProject")

user_prompt = st.number_input("Enter a game rating")
user_prompt2 = st.number_input("Enter the year of release")

w1 = 110212.26226094973
w2 = 7643.05127624
b = -45899.861928343424

if user_prompt != 0 and user_prompt2 != 0:
    response = w1 * user_prompt + b + w2 * user_prompt2

    st.subheader("Response:")
    st.write(response)
