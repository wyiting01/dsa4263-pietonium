import streamlit as st

st.set_page_config(page_title = 'Pietonium Sentiment Prediction App', 
                   page_icon = ":chart_increasing:", 
                   layout = "wide")


# ----- HEADER SECTION -----
# '''
# Welcome user to the app.
# Introduce what this app will help to do.
# '''

with st.container():
    st.title("Welcome to Pietonium Sentiment Prediction App :wave:")
    st.subheader("Need help understanding the sentiments of reviews?")
    st.write("This app will help to output the likely sentiment of any review(s) you input.")


# ----- DIVIDER -----
st.write("---")


# ----- NAVIGATION INSTRUCTIONS -----
# '''
# If a user wants to predict the sentiment of only one review,
# they should navigate to the "Single Review Prediction" page in the sidebar.

# If a user wants to predict the sentiments of multiple reviews,
# they should navigate to the "Multiple Reviews Prediction" page in the sidebar.
# '''

with st.container():
    st.markdown(
    """
    In the sidebar on the left, there are two other pages :
    - Single Review Prediction
    - Multiple Reviews Prediction

    A single review prediction requires you to enter in a review in the text box, while multiple reviews predictions will require a CSV file. 
    
    More instructions can be found in the respective pages to obtain your predictions.
    """)
    st.write("Happy predicting! :nerd_face:")


# ----- DIVIDER -----
st.write("---")


# ----- GITHUB REPO -----
# '''
# This section will help direct people to our github repo
# if they are interested to know how we developed the model, etc.
# '''

with st.container():
    st.write("Curious to know how we train our model? :thinking_face:")
    st.write("Visit our Github Repository to understand what machine learning models we used!")
    st.write("[Github Repo >](https://github.com/wyiting01/dsa4263-pietonium)")


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
