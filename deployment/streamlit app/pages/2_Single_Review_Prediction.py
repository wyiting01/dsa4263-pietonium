import streamlit as st
import pickle
import sys

sys.path.append("../")
from preprocess_fn import noise_entity_removal, mylemmatize, text_normalization, label_to_integer

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
# Parameters needed : text, actual sentiment, predicted sentiment
# """

text, actual_sentiment, predicted_sentiment = "", "", ""

st.write("Let's start predicting your single review!")

# Text box to accept review input
text = st.text_input("Please enter in your review")

col1, _ = st.columns(2)

with col1:
    # Option for user to choose if they have an actual sentiment to compare to
    sentiment_comparison = st.selectbox(
        "Do you know the actual sentiment of the review?",
        ["No", "Yes"]
    )

    # If option is "Yes", choose the correct sentiment
    if sentiment_comparison == "Yes":
        actual_sentiment = st.selectbox(
            "Please choose the sentiment of your review",
            ["Positive", "Negative"]
        )

# ----- PREDICTION -----

# Only show predict button if text box is not empty
if text != "":
    if st.button("Predict Now"):
        st.write("Prediction starting now!")

        # Divider
        st.write("---")

        # Process text and change sentiment to integer
        st.write("Processing of text occurring...")
        processed_text = text_normalization(noise_entity_removal(text))
        processed_actual_sentiment = actual_sentiment
        processed_actual_sentiment = label_to_integer(processed_actual_sentiment.lower())

        # SVM tfidf and model 
        models_meta = {
            "saved_tfidf": "../saved_models/uy_svm1_vectorizer.pkl",
            "saved_model": "../saved_models/uy_svm1.pkl"
            }
        
        st.write("Loading model for prediction...")
        vectorizer = pickle.load(open(models_meta["saved_tfidf"], "rb")) # load saved tfidf vectorizer
        test_x = vectorizer.transform([processed_text])
        saved_model = pickle.load(open(models_meta["saved_model"], "rb")) # load saved model

        # Predict the sentiment
        predicted_sentiment = saved_model.predict(test_x)
        st.write("Prediction done!")

        # Divider
        st.write("---")

        # Print prediction
        if predicted_sentiment == 1:
            predicted_sentiment = 'Positive'
            st.markdown(f"Predicted Sentiment: **:green[Positive]**")
        else:
            predicted_sentiment = 'Negative'
            st.markdown(f"Predicted Sentiment: **:red[Negative]**")

        # Divider
        st.write("---")

        # Check if there was a given sentiment
        if actual_sentiment != "":

            # Check if predicted == actual
            if predicted_sentiment == actual_sentiment:
                st.write("Our prediction matches the sentiment you have given! :tada:")
            else:
                st.write("We are sorry that the prediction does not match the sentiment you have given... :cry:")


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