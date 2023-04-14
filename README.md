<h1>dsa4263-pietonium</h1>

<h2>Folder Structure</h2>

  1. **sentiment_analysis**
    
    1.1 Deep_Learning: fine-tuning experiments of BERT models (files: .py, .ipynb, .txt)
    
    1.2 ML: training experiments of traditional machine learning algorithm, including SVM and XGBoost (files: .py, .ipynb)
    
  2. **topic_modelling**: topic modelling and topic classification experiments
  
  3. **model**: containing model binary files of SVM, XGBoost, LDA, Gensim (except BERT)
  
  4. **data**: raw data and curated data for modelling, with data-preprocessing scripts (files: .csv, .ipynb, .py)
  
  5. **tests**: unit testing experiments
  
  6. **deployment**: streamlit app for scoring api
  
  7. **service (optional)**: dockerized streamlit app
  
  8. **archived**: containing unused files and models for reference, not included in training or inferencing pipeline

<h2>Notes</h2>
  
  - Dockerization of streamlit app is optional because the app can be shared via its own API
  
  - DistilBERT model is too large to push to GitHub (~800MB) and hence be loaded from Hugging Face Hub, which is integrated in code.
