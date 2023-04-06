import streamlit as st
from io import StringIO
import pandas as pd
from preprocess_fn import noise_entity_removal, mylemmatize, text_normalization, label_to_integer

st.set_page_config(page_title = 'Pietonium Sentiment Prediction App', page_icon = ":chart_increasing:", layout = "wide")

# HEADER SECTION
with st.container():
    st.subheader("Welcome to Pietonium Sentiment Prediction App :wave:")
    st.title("Need help understanding the sentiments of reviews?")
    st.write("This app will help to output the likely sentiment of any review(s) you input.")
    st.write("Please follow the steps below to predict your review's sentiment.")

# Put in a divider
st.write("---")

# Allow user to choose if they want to test a sentence or file
with st.container():

    # Option : Single sentence / Multiple sentences
    option = st.selectbox(
        "Would you like to test the prediction on a single review or on multiple reviews?",
        ['A Single Review', 'Multiple Reviews'])
    
    if option == 'A Single Review': # If user select this option, do
        st.text_input("Please enter in your review")

    elif option == 'Multiple Reviews': # If user select this option, do
        uploaded_file = st.file_uploader("Please choose a CSV file")

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




# Put in a divider
st.write("---")

# Direct people to github repo
with st.container():
    st.write("Curious to know how we train our model? :thinking_face:")
    st.write("Visit our Github Repository to understand what machine learning models we used!")
    st.write("[Github Repo >](https://github.com/wyiting01/dsa4263-pietonium)")

# Hide streamlit watermark
hide_streamlit_style = """
                        <style>
                        #MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        </style>
                        """

st.markdown(hide_streamlit_style, unsafe_allow_html = True)