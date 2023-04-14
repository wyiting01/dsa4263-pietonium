<h1>dsa4263-pietonium</h1>

<h2>Repository Structure</h2>

```
├── archive # containing unused files and models for reference, not included in training or inferencing pipeline

├── data
    └── curated/reviews
        └── cleaned_reviews.csv # reviews that have undergone preprocessing
    └── raw
        └── reviews.csv # raw review file
    └── Data Preprocessing.ipynb # jupyter notebook used for preprocessing
    └── Dataset EDA.ipynb # jupyter notebook for EDA on reviews' trend etc.
    └── preprocess_fn.py # python module with preprocess functions
    
 ├── deployment # streamlit app
    └── streamlit_app
    └── comparison.py
    └── distilbert.py
    └── evaluate.py
    └── kl_topic_classification.py
    └── preprocess_fn.py
    └── topic_classification_modularized.py
    
 ├── model # containing model binary files of SVM, XGBoost, LDA, Gensim (except BERT)
      └── lda_gensim
      └── topic_classification
      └── svm.pkl
      └── svm_vectorizer.pkl
      └── uy_svm1.pkl
      └── uy_svm1_vectorizer.pkl
      └── uyen_svm.pkl
      └── uyen_svm_vectorizer.pkl
      └── xgboost.pkl
      └── xgboost_vectorizer.pkl
 
 ├── sentiment_analysis
      └── Deep Learning # fine-tuning experiments of BERT models (files: .py, .ipynb, .txt)
          └── Assess_Performance.ipynb
          └── DistilBERT1_requirements.txt
          └── distilbert.py # python module with distilBert functions for prediction and app deployment
          └── distilbert_train.ipynb
          └── distilbert_train_modules.py # python module with distilBert function for training
      └── ML # training experiments of traditional machine learning algorithm, including SVM and XGBoost (files: .py, .ipynb)
          └── Baseline Model - SVM.ipynb
          └── Baseline Model - XGBoost.ipynb
          └── svm_model.py
          └── xgboost_model.py
          
  ├── service # dockerized streamlit app
      └── requirements.txt
     
  ├── test # unit testing experiments
      └── unit
          └── __init__.py
          └── test_distilbert.py
          └── test_lda_gensim_modularized.py
      └── __init__.py
      
 ├── topic_modelling 
      └── result # contains graphs plot for topic modelling
      └── combine_modularized.py
      └── lda_gensim_modularized.py
      └── lda_gensim_tfidf.py
      └── lda_sklearn.py
      └── presentation_topic_modelling_and_classification.ipynb
      └── requirement.txt
      └── topic_classification_final.py
      └── topic_classification_modularized.py    
```

<h2>Notes</h2>
  
  - Dockerization of streamlit app is optional because the app can be shared via its own API
  
  - DistilBERT model is too large to push to GitHub (~800MB) and hence be loaded from Hugging Face Hub, which is integrated in code.
