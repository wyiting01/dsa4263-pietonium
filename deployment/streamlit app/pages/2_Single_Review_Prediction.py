import streamlit as st

st.set_page_config(page_title = 'Single Review Prediction', 
                   layout = "wide")

# ----- HEADER SECTION -----
with st.container():
    st.subheader("Single Review Prediction")


# ----- PAGE DIVIDER -----
st.write("---")


# ----- DATA INPUT -----
# """
# Ask for the review that user would like to predict.
# Ask if they have an actual sentiment to compare to.
# Predict button.
# """

st.write("Let's start predicting your single review!")

# Text box to accept review input
st.text_input("Please enter in your review")

col1, _ = st.columns(2)

with col1:
    # Option for user to choose if they have an actual sentiment to compare to
    sentiment_comparison = st.selectbox(
        "Do you know the actual sentiment of the review?",
        ["No", "Yes"]
    )

    actual_sentiment = ""

    # If option is "Yes", choose the correct sentiment
    if sentiment_comparison == "Yes":
        actual_sentiment = st.selectbox(
            "Please choose the sentiment of your review",
            ["Positive", "Negative"]
        )

if st.button("Predict Now"):
    st.write("Prediction starting now!")


# ----- HIDING WATERMARK -----
# '''
# For aesthetic purposes, the streamlit watermark is hidden
# '''

hide_streamlit_style = """
                        <style>
                        #MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        </style>
                        """

st.markdown(hide_streamlit_style, unsafe_allow_html = True)