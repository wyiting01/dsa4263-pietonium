import streamlit as st
from io import StringIO
import pandas as pd

st.set_page_config(page_title = 'Multiple Reviews Prediction', 
                   layout = "wide")

# ----- HEADER SECTION -----
with st.container():
    st.subheader("Multiple Reviews Prediction")


# ----- PAGE DIVIDER -----
st.write("---")

# ----- DATA INPUT -----
# """
# Ask for the file of reviews that user would like to predict.
# Ask if the text column name.
# Ask if there are actual sentiments to compare to.
# Ask for sentiment column name if yes.
# Predict button.
# """

st.write("Let's start predicting your reviews!")

# widget to accept uploaded file
uploaded_file = st.file_uploader("Please upload a CSV file", type = "csv")

if uploaded_file is not None:
    # To read file as bytes
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert a string based IO
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # Can be used wherever a "file-like" object is accepted
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

# parameter to know which column in the file is the reviews
TEXT_COL = st.text_input("What is the column name of your reviews? Please input exactly according to the file.")

col1, _ = st.columns(2)

with col1:
    # Option for user to choose if they have an actual sentiment to compare to
    sentiment_comparison = st.selectbox(
        "Do you have the actual sentiment of the reviews? Please input exactly according to the file.",
        ["No", "Yes"]
    )

    ACTUAL_SENTIMENT_COL = ""

    # If option is "Yes", ask for column name
    if sentiment_comparison == "Yes":
        ACTUAL_SENTIMENT_COL = st.text_input("What is the column name of the actual sentiments?")

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