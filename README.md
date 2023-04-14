<h1>dsa4263-pietonium</h1>

<h2>Repository Structure</h2>

```
├── archive

├── data
    └── curated/reviews
        └── ...
    └── raw
        └── ...
    └── ...
    
├── deployment
    └── ...
    
├── model
    └── ...
 
├── sentiment_analysis
    └── Deep Learning # fine-tuning experiments of BERT models (files: .py, .ipynb, .txt)
        └── ...
    └── ML # training experiments of traditional machine learning algorithm, including SVM and XGBoost (files: .py, .ipynb)
        └── ...
          
├── service
    └── ...
     
├── test
    └── ...
      
 ├── topic_modelling 
    └── ...   
```
<h2>Folder Descriptions</h2>

1. archived - contains unused files and models for reference, not included in training or inferencing pipeline
2. data - contains raw and preprocessed reviews
3. deployment - contains modularized and python scripts needed for app deployment and traning-prediction pipeline
4. model - contains binary files of models for sentiment analysis, and topic modelling and classification [SVM, XGBoost, LDA, Gensim (except BERT)]
5. sentiment_analysis - contains jupyter notebooks used for training sentiment analysis pipeline
6. service - contains files needed to dockerise app
7. test - contains python scripts for unit testing
8. topic_modelling - contains jupyter notebook and python scripts for topic modelling and classification

<h2>Notes</h2>
  
  - DistilBERT model is too large to push to GitHub (~800MB) and hence be loaded from Hugging Face Hub, which is integrated in code.
