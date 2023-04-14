import streamlit as st
import pandas as pd
import pickle
import sys

sys.path.append("../")
from distilbert import *

from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ----- CREATE FUNCTIONS -----

def get_sentiments(dataframe):
    model, tokenizer = load_distilbert_model()
    predicted_df = scoring_file_dummy_df(dataframe, model, tokenizer)
    return predicted_df

def get_sentiment(text):
    model, tokenizer = load_distilbert_model()
    predicted_sentiment, sentiment_prob = scoring_single_review(text, model, tokenizer)
    return predicted_sentiment, sentiment_prob

def plot_cm(predicted_labels, actual_labels):
    cm_2d = confusion_matrix(predicted_labels, actual_labels)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.matshow(cm_2d, cmap=plt.cm.Greens, alpha = 0.3)
    for i in range(cm_2d.shape[0]):
        for j in range(cm_2d.shape[1]):
            ax.text(x = j, y = i, s = cm_2d[i, j], va = 'center', ha = 'center')

    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.xlabel('Predictions', fontsize=10)
    plt.ylabel('Actuals', fontsize=10)
    st.pyplot(fig)

st.set_page_config(page_title = 'Sentiment Prediction', 
                   layout = "wide")


# ----- HEADER SECTION -----
with st.container():
    st.subheader("Sentiment Prediction")
    st.write("Please take note that for predictions on multiple reviews, only **:blue[CSV]** files are accepted.")
    st.write("Additionally, the file must have columns **:blue[| Text | Time |]** for the app to run.")


# ----- PAGE DIVIDER -----
st.write("---")


# ----- START OF REVIEW SECTION -----
# Ask if user would like to predict a single / multiple reviews.

st.write("Let's start predicting your review(s)!")


with st.container():
    choice_selection = st.selectbox("Would you like to predict the sentiment of a **:blue[single review]** or of **:blue[multiple reviews]**?",
                                    ["Single Review", "Multiple Reviews"])

single_review, multiple_reviews = st.empty(), st.empty()


# ----- SINGLE REVIEW -----
## If single review
##### Ask for review
##### Ask if they know the sentiment
##### Predict button
##### Run get_sentiment
##### Check if predicted sentiment is the same as given sentiment

if choice_selection == "Single Review":
    review, actual_sentiment, known_sentiment = "", "", ""

    with single_review.container():
        review = st.text_input("What is your review?")

        # Only when a review is given then this part is shown
        if review != "":
            known_sentiment = st.selectbox("Do you know the actual sentiment of the Review?",
                                        ["No", "Yes"])

        else:
            pass
        
        # Only when there is a known sentiment then this part is shown
        if known_sentiment == "Yes":
            actual_sentiment = st.selectbox("What is the actual sentiment?",
                                            ["Positive", "Negative"])
        else:
            pass
        
        # Only when a review is given then predict button will show
        if review != "":
            if st.button("Predict Now"):
                st.write("Prediction starting now!")

                st.write("---")
                st.write("Prediction is now running, please wait patiently. :bow:")
                pred_sentiment, pred_prob = get_sentiment(review)
                st.write("Prediction finished!")

                st.write("---")

                if pred_sentiment == 1:
                    st.write(f"The predicted sentiment is **:green[Positive]**.")
                else:
                    st.write(f"The predicted sentiment is **:red[Negative]**.")

                if actual_sentiment != "":
                    actual_label = label_to_integer_helper(actual_sentiment)

                    if actual_label == pred_sentiment:
                        st.write("Our prediction matches your given sentiment! :tada:")
                    else:
                        st.write("Our prediction did not match your given sentiment. Sorry... :cry:")

                else:
                    pass


# ----- MULTIPLE REVIEWS -----
## If multiple
##### Ask for file
##### Ask if there is an actual sentiment column
##### Predict button
##### Run get_sentiments
##### Plot confusion matrix

if choice_selection == "Multiple Reviews":
    with multiple_reviews.container():
        # widget to accept uploaded file
        uploaded_file = st.file_uploader("Please upload a CSV file", type = "csv")

        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            st.write("File successfully uploaded!")
            st.write(dataframe.head(5))

        known_sentiment = st.selectbox("Do you have a column containing the actual sentiments?",
                                       ["No", "Yes"])

        if known_sentiment == "Yes":
            sentiment_col = st.text_input("What is the name of that column? Please input in the exact name and press **enter** to check its validity.")

            # When sentiment column is given, check if it is in the dataframe
            if sentiment_col != "":
                if sentiment_col in dataframe.columns:
                    st.write(":white_check_mark: :green[That is a valid column!]")

                else:
                    st.write(":x: :red[That is not a valid column, please try again!]")


        # Only when a file is uploaded then the predict button is shown
        if uploaded_file is not None:    
            if st.button("Predict Now"):
                st.write("Prediction starting now!")

                st.write("---")
                st.write("Prediction is now running, please wait patiently. :bow:")
                sample_df = pd.DataFrame(dataframe.head(500)) # Only to run on first 500 reviews due to computer limit
                pred_df = get_sentiments(sample_df) 
                st.write("Prediction finished! Thank you for waiting so patiently!")

                st.write("---")
                st.write("Here's a sneak peek of your first 5 reviews' predicted sentiments. :eyes:")
                st.write(pred_df.head())

                # Only when there is known sentiment then show metrics
                if known_sentiment == "Yes":
                    st.write("---")
                    st.subheader("Here are some evaluation metrics to see how accurate our predictions are!")
                    actual_labels = sample_df[sentiment_col]
                    pred_labels = pred_df['predicted_sentiment']

                    # Convert labels to integer
                    actual_labels = actual_labels.apply(lambda x: label_to_integer_helper(x))
                    pred_labels = pred_labels.apply(lambda x: label_to_integer_helper(x))

                    accuracy = accuracy_score(actual_labels, pred_labels)
                    precision = precision_score(actual_labels, pred_labels)
                    recall = recall_score(actual_labels, pred_labels)
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision.round(2))
                    st.write("Recall: ", recall.round(2))
                    plot_cm(pred_labels, actual_labels)

                else:
                    pass

                # Download button
                predicted_csv = pred_df.to_csv().encode('utf-8')

                st.download_button(
                    label = "Download the predicted sentiments",
                    data = predicted_csv,
                    file_name = "Predicted_Sentiments_of_Review.csv")


# ----- HIDING WATERMARK -----
# For aesthetic purposes, the streamlit watermark is hidden

hide_streamlit_style = """
                        <style>
                        #MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        </style>
                        """

st.markdown(hide_streamlit_style, unsafe_allow_html = True)