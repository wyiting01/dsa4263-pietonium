import streamlit as st
import pandas as pd
import pickle
import sys

sys.path.append("../")
from preprocess_fn import *

sys.path.append("../../sentiment_analysis/Deep_Learning/")
from distilbert import *

from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ----- CREATE FUNCTIONS -----

def get_sentiments(dataframe):
    model, tokenizer = load_distilbert_model()
    predicted_df = scoring_file_thread(dataframe, model, tokenizer)
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
    st.write("Please take note...")


# ----- PAGE DIVIDER -----
st.write("---")


# ----- START OF REVIEW SECTION -----
# Ask if user would like to predict a single / multiple reviews.

st.write("Let's start predicting your review(s)!")


with st.container():
    choice_selection = st.selectbox("Would you like to predict the sentiment of a **single review** or of **multiple reviews**?",
                                    ["", "Single Review", "Multiple Reviews"])

single_review, multiple_reviews = st.empty(), st.empty()


# ----- SINGLE REVIEW -----
## if single review
##### ask for review
##### ask if they know the sentiment
##### predict button
##### run get_sentiment
##### check if predicted sentiment is the same as given sentiment

if choice_selection == "Single Review":
    review, actual_sentiment = "", ""

    with single_review.container():
        review = st.text_input("What is your review?")

        known_sentiment = st.selectbox("Do you know the actual sentiment of the Review?",
                                       ["No", "Yes"])
        
        if known_sentiment == "Yes":
            actual_sentiment = st.selectbox("What is the actual sentiment?",
                                            ["Positive", "Negative"])
        else:
            pass

        if review != "":
            if st.button("Predict Now"):
                st.write("Prediction starting now!")

                st.write("---")
                st.write("Prediction is now running, please wait patiently. Thank you!")
                pred_sentiment, pred_prob = get_sentiment(review)
                st.write("Prediction finished!")

                st.write("---")

                if pred_sentiment == 1:
                    st.write(f"The predicted sentiment is **:green[Positive]**")
                else:
                    st.write(f"The predicted sentiment is **:red[Negative]**")

                if actual_sentiment != "":
                    actual_label = label_to_integer_helper(actual_sentiment)

                    if actual_label == pred_sentiment:
                        st.write("Our prediction matches your sentiment! :tada:")
                    else:
                        st.write("Our prediction did not match you sentiment. Sorry... :cry:")

                else:
                    pass

    if st.button("Reset"):
        single_review.empty()


# ----- MULTIPLE REVIEWS -----
## if multiple
##### ask for file
##### predict button
##### run get_sentiments
##### plot confusion matrix

if choice_selection == "Multiple Reviews":
    with multiple_reviews.container():
        # widget to accept uploaded file
        uploaded_file = st.file_uploader("Please upload a CSV file", type = "csv")

        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            st.write("File successfully uploaded!")
            st.write(dataframe.head(5))

            if st.button("Predict Now"):
                st.write("Prediction starting now!")

                st.write("---")
                st.write("Prediction is now running, please wait patiently. Thank you!")
                pred_df = get_sentiments(uploaded_file)
                st.write("Prediction finished! Thank you for waiting so patiently!")

                st.write("---")
                st.write(pred_df)

    if st.button("Reset"):
        multiple_reviews.empty()           
    


# # ----- COLUMN NAMES -----

# # Obtain reviews column
# TEXT_COL = st.text_input("What is the column name of your reviews? Please input exactly according to the file.")

# if TEXT_COL != "":

#     # Test to see if a valid column name was given
#     try:
#         if TEXT_COL in dataframe.columns:
#             st.markdown(":green[That is a valid column name]")
#     except:
#         st.markdown(":red[Error: It seems that an invalid column name was given. Please check your column name again.]")

# col1, _ = st.columns(2)

# with col1:
#     # Option for user to choose if they have an actual sentiment to compare to
#     sentiment_comparison = st.selectbox(
#         "Do you have the actual sentiment of the reviews?",
#         ["No", "Yes"]
#     )

# # If option is "Yes", obtain sentiment column name
# if sentiment_comparison == "Yes":
#     ACTUAL_SENTIMENT_COL = st.text_input("What is the column name of the actual sentiments? Please input exactly according to the file.")

#     if ACTUAL_SENTIMENT_COL != "":

#         # Test to see if a valid column name is given
#         try:
#             if ACTUAL_SENTIMENT_COL in dataframe.columns:
#                 st.markdown(":green[That is a valid column name]")
#         except:
#             st.markdown(":red[Error: It seems that an invalid column name was given. Please check your column name again.]")


# # ----- PREDICTION -----

# if st.button("Predict Now"):
#     st.write("Prediction starting now!")

#     # Divider
#     st.write("---")

#     # Reviews and labels to be processed
#     st.write("Processing of review occuring...")
#     preprocessed_df = preprocess(dataframe, text_col_name = TEXT_COL, label_col_name = ACTUAL_SENTIMENT_COL)
#     preprocessed_text = preprocessed_df[TEXT_COL].values.tolist()
#     st.write("Review successfully processed!")

#     # SVM tfidf and model 
#     models_meta = {
#         "saved_tfidf": "../saved_models/uy_svm1_vectorizer.pkl",
#         "saved_model": "../saved_models/uy_svm1.pkl"
#         }
        
#     st.write("Loading model for prediction...")
#     vectorizer = pickle.load(open(models_meta["saved_tfidf"], "rb")) # load saved tfidf vectorizer
#     saved_model = pickle.load(open(models_meta["saved_model"], "rb")) # load saved model
#     test_x = vectorizer.transform(preprocessed_text)

#     # Predict the sentiment
#     st.write("Prediction in progress...")
#     predicted_y = saved_model.predict(test_x).tolist()

#     # Convert integer sentiment to string
#     predicted_y_str = []
#     for p in predicted_y:
#         predicted_y_str.append(integer_to_label(p))

#     st.write("Prediction done!")

#     # Divider
#     st.write("---")
    
#     # If there is sentiment to compare to...
#     if sentiment_comparison == "Yes":

#         st.subheader("Here are some metrics when comparing our predictions to the actual sentiments...")

#         # Process the original sentiments
#         preprocessed_labels = preprocessed_df[ACTUAL_SENTIMENT_COL].values.tolist() 

#         # Output the accuracy, precision and recall score
#         accuracy = accuracy_score(preprocessed_labels, predicted_y)
#         precision = precision_score(preprocessed_labels, predicted_y)
#         recall = recall_score(preprocessed_labels, predicted_y)
#         st.write("Accuracy: ", accuracy.round(2))
#         st.write("Precision: ", precision.round(2))
#         st.write("Recall: ", recall.round(2))

#         # Plot confusion matrix
#         st.subheader("Confusion Matrix")
#         cm_2d = confusion_matrix(preprocessed_labels, predicted_y)
#         fig, ax = plt.subplots(figsize=(2, 2))
#         ax.matshow(cm_2d, cmap=plt.cm.Greens, alpha = 0.3)
#         for i in range(cm_2d.shape[0]):
#             for j in range(cm_2d.shape[1]):
#                 ax.text(x = j, y = i, s = cm_2d[i, j], va = 'center', ha = 'center')

#         plt.xticks([0, 1], ['Negative', 'Positive'])
#         plt.yticks([0, 1], ['Negative', 'Positive'])
#         plt.xlabel('Predictions', fontsize=10)
#         plt.ylabel('Actuals', fontsize=10)
#         st.pyplot(fig)

#         # Create a new dataframe to output
#         predicted_data = {
#             TEXT_COL: dataframe[TEXT_COL].values.tolist(),
#             "Original Sentiments": dataframe[ACTUAL_SENTIMENT_COL].values.tolist(),
#             "Predicted Sentiments": predicted_y_str
#         }

#         predicted_df = pd.DataFrame(predicted_data)
#         st.write(predicted_df.head())

#         # Download button
#         predicted_csv = predicted_df.to_csv().encode('utf-8')

#         st.download_button(
#             label = "Download the predicted sentiments",
#             data = predicted_csv,
#             file_name = "Predicted_Sentiments_of_Review.csv",
#             mime = "text/csv")
    
#     else: # No comparison to be done
#         predicted_data = {
#             TEXT_COL: dataframe[TEXT_COL].values.tolist(),
#             "Predicted Sentiments": predicted_y_str
#         }

#         predicted_df = pd.DataFrame(predicted_data)
#         st.write(predicted_df.head())

#         # Download button
#         predicted_csv = predicted_df.to_csv().encode('utf-8')

#         st.download_button(
#             label = "Download the predicted sentiments",
#             data = predicted_csv,
#             file_name = "Predicted_Sentiments_of_Review.csv")


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