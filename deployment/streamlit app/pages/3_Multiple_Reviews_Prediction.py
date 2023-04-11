import streamlit as st
import pandas as pd
import pickle
import sys

sys.path.append("../")
from preprocess_fn import noise_entity_removal, mylemmatize, text_normalization, label_to_integer, integer_to_label, preprocess
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from io import BytesIO

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
        if TEXT_COL in dataframe.columns:
            st.markdown(":green[That is a valid column name]")
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

    if ACTUAL_SENTIMENT_COL != "":

        # Test to see if a valid column name is given
        try:
            if ACTUAL_SENTIMENT_COL in dataframe.columns:
                st.markdown(":green[That is a valid column name]")
        except:
            st.markdown(":red[Error: It seems that an invalid column name was given. Please check your column name again.]")


# ----- PREDICTION -----

if st.button("Predict Now"):
    st.write("Prediction starting now!")

    # Divider
    st.write("---")

    # Reviews and labels to be processed
    st.write("Processing of review occuring...")
    preprocessed_df = preprocess(dataframe, text_col_name = TEXT_COL, label_col_name = ACTUAL_SENTIMENT_COL)
    preprocessed_text = preprocessed_df[TEXT_COL].values.tolist()
    st.write("Review successfully processed!")

    # SVM tfidf and model 
    models_meta = {
        "saved_tfidf": "../saved_models/uy_svm1_vectorizer.pkl",
        "saved_model": "../saved_models/uy_svm1.pkl"
        }
        
    st.write("Loading model for prediction...")
    vectorizer = pickle.load(open(models_meta["saved_tfidf"], "rb")) # load saved tfidf vectorizer
    saved_model = pickle.load(open(models_meta["saved_model"], "rb")) # load saved model
    test_x = vectorizer.transform(preprocessed_text)

    # Predict the sentiment
    st.write("Prediction in progress...")
    predicted_y = saved_model.predict(test_x).tolist()

    # Convert integer sentiment to string
    predicted_y_str = []
    for p in predicted_y:
        predicted_y_str.append(integer_to_label(p))

    st.write("Prediction done!")

    # Divider
    st.write("---")
    
    # If there is sentiment to compare to...
    if sentiment_comparison == "Yes":

        st.subheader("Here are some metrics when comparing our predictions to the actual sentiments...")

        # Process the original sentiments
        preprocessed_labels = preprocessed_df[ACTUAL_SENTIMENT_COL].values.tolist() 

        # Output the accuracy, precision and recall score
        accuracy = accuracy_score(preprocessed_labels, predicted_y)
        precision = precision_score(preprocessed_labels, predicted_y)
        recall = recall_score(preprocessed_labels, predicted_y)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision.round(2))
        st.write("Recall: ", recall.round(2))

        # Plot confusion matrix
        st.subheader("Confusion Matrix")
        cm_2d = confusion_matrix(preprocessed_labels, predicted_y)
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

        # Create a new dataframe to output
        predicted_data = {
            TEXT_COL: dataframe[TEXT_COL].values.tolist(),
            "Original Sentiments": dataframe[ACTUAL_SENTIMENT_COL].values.tolist(),
            "Predicted Sentiments": predicted_y_str
        }

        predicted_df = pd.DataFrame(predicted_data)
        st.write(predicted_df.head())

        # Download button
        predicted_csv = predicted_df.to_csv().encode('utf-8')

        st.download_button(
            label = "Download the predicted sentiments",
            data = predicted_csv,
            file_name = "Predicted_Sentiments_of_Review.csv",
            mime = "text/csv")
    
    else: # No comparison to be done
        predicted_data = {
            TEXT_COL: dataframe[TEXT_COL].values.tolist(),
            "Predicted Sentiments": predicted_y_str
        }

        predicted_df = pd.DataFrame(predicted_data)
        st.write(predicted_df.head())

        # Download button
        predicted_csv = predicted_df.to_csv().encode('utf-8')

        st.download_button(
            label = "Download the predicted sentiments",
            data = predicted_csv,
            file_name = "Predicted_Sentiments_of_Review.csv")





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