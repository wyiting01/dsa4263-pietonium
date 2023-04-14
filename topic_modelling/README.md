# Folder Structure
```
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
1. result folder - consists of plots, csv, pyldavis from topic modelling and classification using LDA(Gensim)
2. combine_modularized.py - combination of scripts of topic_classification_modularized.py and lda_gensim_modularized.py for unit testing purposes
3. lda_gensim_modularized.py - modularized script of topic modelling using LDA(Sklearn)
4. lda_gensim_tfidf.py - topic modelling using LDA(Gensim)
5. lda_sklearn.py - modularized script of topic modelling using LDA(Sklearn) and XGboost for topic classification
6. presentation_topic_modelling_and_classification.ipynb - presentation pipeline for training and prediction
7. requirement.txt - requirements file to run scripts involving LDA(Gensim)
8. topic_classification_final.py - topic classification that make uses of LDA(Gensim)'s topic modelling result
9. topic_classification_modularized.py - modularized script of topic classification 


