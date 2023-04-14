# Folder Structure
```
├── deployment
    └── streamlit_app
        └── 1_Home_page.py
        └── pages
            └── 2_Sentiment_Prediction.py
            └── 3_Topic_Classification.py
    └── comparison.py
    └── distilbert.py
    └── evaluate.py
    └── preprocess_fn.py
    └── topic_classification_modularized.py
```

<h2>File/Folder Descriptions</h2>
1. streamlit_app - consists of python scripts used to create the different pages of the app
2. comparison - modularized script used to compare different models
3. distilbert - modularized script to prepare, train and predict data for distilBert
4. evaluate - modularize script used to make predictions on the test data when multiple models are given
5. preprocess_fn - modularized script with functions needed to process raw text
6. topic_classification_modularize - modularized script to prepare, train and predict data for topic modelling and classification
