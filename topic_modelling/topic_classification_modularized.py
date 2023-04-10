import gensim
from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def read_split_data(path):
    """
    Read labelled data obtained from previous LDA topic modelling
    and split train and test in ratio 0.2
    """
    processed_data = pd.read_csv(path)
    train , test = train_test_split(processed_data, test_size = 0.2, random_state = 42)
    return (train, test)

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

def preprocess(data_inv, id2words):
    """
    1. Combine all reviews into list
    2. Split sentences to individual words
    3. Build bigram and trigram models
    4. Form bigram and trigram
    5. Create corpus
    """
    data_invs = data_inv.Text.values.tolist()

    data_words = split_sentence(data_invs)

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

def load_lda_model():
    """
    Load LDA-tfidf model from topic modelling
    """
    lda_tfidf_model = gensim.models.LdaMulticore.load('../model/kl_lda_tfidf_model.pkl')
    return lda_tfidf_model

def load_model_dictionary():
    """
    Load LDA-tfidf dictionary from topic modelling
    """    
    dictionary = corpora.Dictionary.load('../model/kl_lda_tfidf_model.pkl.id2word')
    return dictionary

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

   
def final_xgb():
    """
    Load final xgb model
    """
    xgb_model_loaded = pickle.load(open('../model/xgb_topic_classification_final.pkl', "rb"))
    return xgb_model_loaded

def baseline_xgb():
    """
    Load Baseline xgb model
    """
    xgb_model_loaded = pickle.load(open('../model/xgb_topic_classification_base.pkl', "rb"))
    return xgb_model_loaded

def predict_and_evaluate_model(chosen_model, x_test_scale, y_test):
    """
    1. Predict topic number of new data
    2. Produce Classification report 
    3. Produce Confusion matrix
    """
    y_predicted_algo = chosen_model.predict(x_test_scale)

    report = classification_report(y_test, y_predicted_algo, output_dict=True,  zero_division=0)
    classfication_df = pd.DataFrame(report).transpose()
    
    cm = confusion_matrix(y_test,  y_predicted_algo)
    confusion_matrix_df = pd.DataFrame(cm)
    return (classfication_df, confusion_matrix_df)