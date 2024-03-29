import gensim
from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from nltk import FreqDist
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle

def load_lda_model(path):
    """
    load lda_tfidf model 
    '/model/lda_gensim/lda_tfidf_model_FINAL.pkl'
    """
    lda_tfidf_model = gensim.models.LdaMulticore.load(path)
    
    return lda_tfidf_model

def load_id2word(path):
    """
    load lda_tfidf dictionary
    '../model/lda_gensim/lda_tfidf_model_FINAL.pkl.id2word'
    """
    dictionary = corpora.Dictionary.load(path)
    return dictionary

def load_labelled_csv(path):
    """
    Read in review that has labelled topic which was done by LDA
    'topic_modelling/result/dominant_topic_in_each_sentence.csv'
    """
    processed_data = pd.read_csv(path)
    return processed_data

def load_base_svc(path):
    """
    Load Base SVC model
    'model/topic_classification/svm_topic_classification_basic.pkl'
    """
    base_svc = pickle.load(open(path, 'rb'))
    return base_svc

def load_final_svc(path):
    """
    Load Final SVC model
    'model/topic_classification/svm_topic_classification_final.pkl'
    """
    final_svc = pickle.load(open(path, 'rb'))
    return final_svc

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

def convert_vector_to_scaled_array(x, y = None):
    """
    For x data: Convert vectors to numpy array and scale
    For y data: Convert to numpy array
    """
    scaler = StandardScaler()

    x_arr = np.array(x)
    x_arr_scale = scaler.fit_transform(x_arr)

    if y != None:
        y_arr = np.array(y)
        y_arr_scale = scaler.fit_transform(y_arr)
        return (x_arr_scale, y_arr_scale)
    
    else:
        return x_arr_scale

def tune_hyperparameter(x_train, y_train):
    """
    Obtain the optimal C, gamma number and kernel
    """
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(SVC(random_state=1),param_grid,refit=True,verbose=2)
    grid_result = grid.fit(x_train, y_train)
    print(grid_result.best_params_)
   

def predict_and_evaluate_model(chosen_model, x_test_scale, y_test = None):
    """
    1. Predict test y label
    2. Produce Classification report 
    3. Produce Confusion matrix
    """
    y_predicted_algo = chosen_model.predict(x_test_scale)

    if y_test != None:
        report = classification_report(y_test, y_predicted_algo, output_dict=True,  zero_division=0)
        classfication_df = pd.DataFrame(report).transpose()
        cm = confusion_matrix(y_test, y_predicted_algo)
        confusion_matrix_df = pd.DataFrame(cm)
        return (classfication_df, confusion_matrix_df)
    else:
        return y_predicted_algo

def baseline_svc(x_train_scale, y_train):
    svm_tfidf = SVC(random_state= 1, C = 10, gamma = 10, kernel='sigmoid', decision_function_shape='ovo').fit(x_train_scale, y_train)
    return svm_tfidf

def final_svc (x_train_scale, y_train, c_num, gamma_num, kernel_name):
    svm_tfidf_final = SVC(random_state= 1, C = c_num, gamma = gamma_num, kernel= kernel_name, decision_function_shape='ovo').fit(x_train_scale, y_train)
    return svm_tfidf_final
