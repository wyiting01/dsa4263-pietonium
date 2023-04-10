LDA,TF-IDF

- kl_lda_bow.py : basic LDA model using BOW
- kl_lda_tfidf.py: LDA model using Tfidf 
- kl_topic_classification.py: Using Tfidf vector from LDA model (kl_lda_tfidf.py) and feed into 3 machine learning (Logistic Regression, Naives Bayes and Logistic Regression with mini batch SGD) 


LDA (sklearn).ipynb -> notebook presentation & workings with some plots & visualisations 
Lda_sklearn.py -> script with functions defined for module exports

lda_sklearn pipeline:
df -> vectorize -> tfidf -> lda_model -> fine tuning -> topic classification -> evaluation (XGboost & SVM)

Requirements:
pandas
numpy
matplotlib.pyplot 
nltk
scikit-learn
gensim

