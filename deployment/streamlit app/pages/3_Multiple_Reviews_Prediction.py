import streamlit as st
import pandas as pd
import pickle
import sys

sys.path.append("../")
from preprocess_fn import noise_entity_removal, mylemmatize, text_normalization, label_to_integer

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

TEXT_COL, ACTUAL_SENTIMENT_COL = "", ""

st.write("Let's start predicting your reviews!")

# widget to accept uploaded file
uploaded_file = st.file_uploader("Please upload a CSV file", type = "csv")

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write("File successfully uploaded!")
    st.write(dataframe.head(5))


# ----- COLUMN NAMES -----

# Obtain reviews column
TEXT_COL = st.text_input("What is the column name of your reviews? Please input exactly according to the file.")

if TEXT_COL != "":

    # Test to see if a valid column name was given
    try:
        st.write("The first 5 reviews of your file are")
        st.write(dataframe[TEXT_COL].head(5))
    except:
        st.markdown(":red[Error: It seems that an invalid column name was given. Please check your column name again.]")

col1, _ = st.columns(2)

with col1:
    # Option for user to choose if they have an actual sentiment to compare to
    sentiment_comparison = st.selectbox(
        "Do you have the actual sentiment of the reviews?",
        ["No", "Yes"]
    )

# If option is "Yes", obtain sentiment column name
if sentiment_comparison == "Yes":
    ACTUAL_SENTIMENT_COL = st.text_input("What is the column name of the actual sentiments? Please input exactly according to the file.")

    try:
        st.write("The first 5 sentiments of your file are")
        st.write(dataframe[ACTUAL_SENTIMENT_COL].head(5))
    except:
        st.markdown(":red[Error: It seems that an invalid column name was given. Please check your column name again.]")


# ----- PREDICTION -----

if st.button("Predict Now"):
    st.write("Prediction starting now!")

    # Divider
    st.write("---")

    # Reviews and labels to be processed
    

    # Process text and change sentiment to integer
    st.write("Processing of text occurring...")
    processed_text = text_normalization(noise_entity_removal(text))
    processed_actual_sentiment = actual_sentiment
    processed_actual_sentiment = label_to_integer(processed_actual_sentiment.lower())


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