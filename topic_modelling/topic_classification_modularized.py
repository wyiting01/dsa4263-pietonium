import gensim
from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
from xgboost import XGBClassifier
from nltk import FreqDist
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def load_lda_model():
    """
    load lda_tfidf model 
    """
    lda_tfidf_model = gensim.models.LdaMulticore.load('../model/lda_gensim/lda_tfidf_model_FINAL.pkl')
    
    return lda_tfidf_model

def load_id2word():
    """
    load lda_tfidf dictionary
    """
    dictionary = corpora.Dictionary.load('../model/lda_gensim/lda_tfidf_model_FINAL.pkl.id2word')
    return dictionary

def load_labelled_csv():
    """
    Read in review that has labelled topic which was done by LDA
    """
    processed_data = pd.read_csv('./result/dominant_topic_in_each_sentence.csv')
    return processed_data

def split_test_train(processed_data):
    """
    Read labelled data obtained from previous LDA topic modelling
    and split train and test in ratio 0.2
    """
    train , test = train_test_split(processed_data, test_size = 0.2, random_state = 42)
    return (train, test)

# get the top 10 highest count words across all reviews
def get_top_10_words(data_inv):
    lst = []
    for i in range(len(data_inv)):
        for k in data_inv[i]:
            lst.append(k)
    fdist = FreqDist(lst) # a frequency distribution of words (word count over the corpus)
    top_k_words, _ = zip(*fdist.most_common(10)) # unzip the words and word count tuples
    return(list(top_k_words)) 

# remove the top 10 highest count words from all reviews
def remove_top_10_words(data_inv, top_10_lst):
    lst = []
    for i in range(len(data_inv)):
        sentence = []
        for k in data_inv[i]:
            if k not in top_10_lst:
                sentence.append(k)
        lst.append(sentence)
    return lst

def y_label(train,test):
    """
    Obtain the topic number of each reviews in test and train respectively
    """
    y_train = train['Dominant_Topic']
    y_test = test['Dominant_Topic']
    return (y_train, y_test)

def split_sentence(text):
    """
    Split a sentence into a list 
    """
    lst = []
    for sentence in text:
        lst.append(sentence.split())
    return lst

def preprocess_test_train(data_inv, id2words):
    """
    1. Combine all reviews into list
    2. Split sentences to individual words
    3. Removed top 10 highest frequency count words
    3. Build bigram and trigram models
    4. Form bigram and trigram
    5. Create corpus
    """
    data_invs = data_inv.Text.values.tolist()

    data_words = split_sentence(data_invs)
    remove_lst = get_top_10_words(data_words)
    data_words = remove_top_10_words(data_words,remove_lst)

    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)  

    data_bigrams = make_bigrams(data_words, bigram_phrases)
    data_bigrams_trigrams = make_trigrams(data_bigrams, bigram_phrases, trigram_phrases)

    collection_texts = data_bigrams_trigrams
    bow_corpuss = [id2words.doc2bow(text) for text in collection_texts]
    return (bow_corpuss)

def make_bigrams(texts, bigram_text):
    """
    Faster way to detect bigrams
    """
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_text, trigram_text,):
    """
    Faster way to detect bigrams/trigrams
    """
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    trigram_mod = gensim.models.phrases.Phraser(trigram_text)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]    

def create_vectors(corpuss, data, lda_model, k):
    """
    1. Create tfidf model and get tfidf corpus
    2. Convert tfidf corpus to vector in order to feed it into classification models
    """
    tfidf_corpus = models.TfidfModel(corpuss, smartirs='ntc')
    corpus_tfidf = tfidf_corpus[corpuss]
    vecs_lst = []
    for i in range(len(data)):
        top_topics = lda_model.get_document_topics(corpus_tfidf[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(k)]
        vecs_lst.append(topic_vec)
    return (vecs_lst)

def convert_vector_to_scaled_array(train_vecs, y_train, test_vecs, y_test):
    """
    For x data: Convert vectors to numpy array and scale
    For y data: Convert to numpy array
    """
    x_train = np.array(train_vecs)
    y_train = np.array(y_train)
    x_test = np.array(test_vecs)
    y_test = np.array(y_test)

    scaler = StandardScaler()
    x_train_scale = scaler.fit_transform(x_train)
    x_test_scale = scaler.fit_transform(x_test)

    return (x_train_scale, y_train, x_test_scale, y_test)

def baseline_xgb(x_train_scale, y_train):
    """
    Run Baseline xgb model
    """
    xgbc_base = XGBClassifier(n_estimators= 100 , seed = 27)
    # Fit train data
    xgbc_tfidf_base = xgbc_base.fit(x_train_scale, y_train)
    return xgbc_tfidf_base

def tune_hyperparameter(x_train_scale, y_train):
    """
    Obtain the optimal n_estimators
    """
    model = XGBClassifier()
    n_estimators = range(50, 1000, 50)
    param_grid = dict(n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x_train_scale,y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def final_xgb(x_train_scale, y_train):
    """
    Run Final xgb model
    """
    xgbc = XGBClassifier(n_estimators= 50 , seed = 27)
    # Fit train data
    xgbc_tfidf = xgbc.fit(x_train_scale, y_train)
    return xgbc_tfidf

def predict_and_evaluate_model(chosen_model, x_test_scale, y_test):
    """
    1. Predict test y label
    2. Produce Classification report 
    3. Produce Confusion matrix
    """
    y_predicted_algo = chosen_model.predict(x_test_scale)

    report = classification_report(y_test, y_predicted_algo, output_dict=True,  zero_division=0)
    classfication_df = pd.DataFrame(report).transpose()
    
    cm = confusion_matrix(y_test, y_predicted_algo)
    confusion_matrix_df = pd.DataFrame(cm)
    return (classfication_df, confusion_matrix_df)