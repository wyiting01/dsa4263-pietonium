import streamlit as st
import pickle
import sys

sys.path.append("../")
from preprocess_fn import *

sys.path.append("../")
from topic_classification_modularized import *

st.set_page_config(page_title = 'Topic Classification', 
                   layout = "wide")

# ----- CREATE FUNCTIONS -----

def get_topics(dataframe):
    """
    This function will get the topic of multiple reviews.
    """

    topic_dict = {
        0: "Dessert",
        1: "Snacks and Chips",
        2: "Pet Related"
    }

    # lda_path = "../../model/lda_gensim/lda_tfidf_model_FINAL.pkl"
    # id2word_path = "../../model/lda_gensim/lda_tfidf_model_FINAL.pkl.id2word"
    # final_svc_path = "../../model/topic_classification/svm_topic_classification_deployment.pkl"
    lda_path = "lda_tfidf_model_FINAL.pkl"
    id2word_path = "lda_tfidf_model_FINAL.pkl.id2word"
    final_svc_path = "svm_topic_classification_deployment.pkl"
    lda_model = load_lda_model(lda_path)
    id2word = load_id2word(id2word_path)
    final_svc = load_final_svc(final_svc_path)

    df_preprocessed = pd.DataFrame()
    df_preprocessed['Text'] = dataframe['Text'].apply(lambda x: text_normalization(noise_entity_removal(x)))
    x_corpus = preprocess_test_train(df_preprocessed, id2word)
    x_vectors = create_vectors(x_corpus, df_preprocessed, lda_model, 3)
    x_vec_scale = convert_vector_to_scaled_array(x_vectors)
    topic_label = final_svc.predict(x_vec_scale)
    topics = []
    for label in topic_label:
        topics.append(topic_dict.get(label))

    classify_df = dataframe.copy()
    classify_df['Topic'] = topics

    return classify_df


# ----- HEADER SECTION -----
with st.container():
    st.subheader("Topic Classification")


# ----- PAGE DIVIDER -----
st.write("---")


# ----- START OF REVIEW SECTION -----
# Ask if user would like to predict a single / multiple reviews.

st.write("Let's start classifying your review(s)!")

with st.container():
    choice_selection = st.selectbox("Would you like to classify a **:blue[single review]** or of **:blue[multiple reviews]**?",
                                    ["Single Review", "Multiple Reviews"])

single_review, multiple_reviews = st.empty(), st.empty()


# ----- SINGLE REVIEW -----
## If single review
##### Ask for review
##### Classify button
##### Run get_topic

if choice_selection == "Single Review":
    with single_review.container():
        review = st.text_input("What is your review?")

        # Only when a review is given then this part is shown
        if review != "":
            review_df = pd.DataFrame([review], columns = ['Text'])
            if st.button("Classify Now"):
                st.write("Classification starting now!")

                st.write("---")
                st.write("Classification is now running, please wait patiently. :bow:")
                classified_df = get_topics(review_df)
                classified_topic = classified_df['Topic'].values[0]
                st.write("Classification finished!")

                st.write("---")
                st.write(f"The topic of your reviews is : **:blue[{classified_topic}]**")


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

            if st.button("Classify Now"):
                st.write("Classification starting now!")

                st.write("---")
                st.write("Classification is now running, please wait patiently. :bow:")
                classified_df = get_topics(dataframe)
                st.write("Classification finished!")

                st.write("---")
                st.write("Here's a sneak peek of your first 5 review's topic! :eyes:")
                st.write(classified_df.head())

                # Download button
                classified_csv = classified_df.to_csv().encode('utf-8')

                st.download_button(
                    label = "Download the classified topics",
                    data = classified_csv,
                    file_name = "Classified_Topics_of_Review.csv")


# ----- HIDING WATERMARK -----
# For aesthetic purposes, the streamlit watermark is hidden

hide_streamlit_style = """
                        <style>
                        #MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        </style>
                        """

st.markdown(hide_streamlit_style, unsafe_allow_html = True)