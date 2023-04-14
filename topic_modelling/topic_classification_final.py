# Topic Classification using TFIDF (obtained from LDA using TFIDF (Gensim))
# adapted from https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28
import gensim
from gensim import corpora, models
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
from nltk import FreqDist
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# split a sentence into a list 
def split_sentence(text):
    lst = []
    for sentence in text:
        lst.append(sentence.split())
    return lst

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

def preprocess(data_inv, id2words):
    # split sentences to individual words
    data_words = split_sentence(data_inv)
    remove_lst = get_top_10_words(data_words)
    data_words = remove_top_10_words(data_words,remove_lst)

    # Build the bigram and trigram models
    # bigrams are two words frequently occur together
    # trigrams are three words frequently occurring 
    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)  

    # form bigrams
    data_bigrams = make_bigrams(data_words, bigram_phrases)

    # form trigrams
    data_bigrams_trigrams = make_trigrams(data_bigrams, bigram_phrases, trigram_phrases)

    # Create Corpus
    collection_texts = data_bigrams_trigrams

    # Term Document Frequency
    # convert list of words into bag-of-words format
    bow_corpuss = [id2words.doc2bow(text) for text in collection_texts]
    return (bow_corpuss)


def make_bigrams(texts, bigram_text):
    # faster way to detect bigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_text, trigram_text,):
    # faster way to detect bigrams/trigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    trigram_mod = gensim.models.phrases.Phraser(trigram_text)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]    

# save results 
def save_result(path, y_test_algo, y_predicted_algo):
    # generate classification report
    report = classification_report(y_test_algo, y_predicted_algo, output_dict=True,  zero_division=0)
    # convert classification report to dataframe 
    classfication_df = pd.DataFrame(report).transpose()
    # generate confusion matrix 
    cm = confusion_matrix(y_test_algo,  y_predicted_algo)
    # convert from matrix to dataframe
    confusion_matrix_df = pd.DataFrame(cm)
    # save both confusion matrix and classification report into one csv file
    """
    with open(path,'a') as f:
        for df in [classfication_df, confusion_matrix_df]:
            df.to_csv(f)
            f.write("\n")
    """
    return (classfication_df, confusion_matrix_df)

# convert corpus to vector in order to feed it into sklearn models
def create_vectors(corpuss, data, lda_model, k):
    # create tf-idf model
    tfidf_corpus = models.TfidfModel(corpuss, smartirs='ntc')
    corpus_tfidf = tfidf_corpus[corpuss]
    vecs_lst = []
    # get the feature vectors for every review
    for i in range(len(data)):
        # capture the instances where a review is presented with 0% in some topics, and the representation for each review will add up to 100%.
        top_topics = lda_model.get_document_topics(corpus_tfidf[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(k)]
        vecs_lst.append(topic_vec)
    return (vecs_lst)

if __name__ == '__main__':
    # load lda_tfidf model 
    lda_tfidf_model = gensim.models.LdaMulticore.load('../model/lda_gensim/lda_tfidf_model_FINAL.pkl')
    
    # load lda_tfidf dictionary
    dictionary = corpora.Dictionary.load('../model/lda_gensim/lda_tfidf_model_FINAL.pkl.id2word')

    # read in review that has labelled topic which was done by LDA
    processed_data = pd.read_csv('./result/dominant_topic_in_each_sentence.csv')

    # split train and test in ratio 0.2
    train,test = train_test_split(processed_data, test_size = 0.2, random_state = 42)
   
    y_train = train['Dominant_Topic']
    y_test = test['Dominant_Topic']

    # pre-process the train and test data to get respective corpus
    train_data = train.Text.values.tolist()
    test_data = test.Text.values.tolist()
    x_train_corpus = preprocess(train_data, dictionary) 
    x_test_corpus = preprocess(test_data, dictionary)
    
    # 3 is the optimal number of topics that LDA using Tfidf choses
    train_vecs = create_vectors(x_train_corpus, train, lda_tfidf_model, 3)
    test_vecs = create_vectors(x_test_corpus, test, lda_tfidf_model, 3)
    
    # convert to numpy array 
    x_train = np.array(train_vecs)
    y_train = np.array(y_train)
    x_test = np.array(test_vecs)
    y_test = np.array(y_test)
    
    # Scale Data
    scaler = StandardScaler()
    x_train_scale = scaler.fit_transform(x_train)
    x_test_scale = scaler.fit_transform(x_test)

    # create directory to keep results
    os.makedirs('result/', exist_ok=True)
    os.makedirs('../model/topic_classification/', exist_ok=True)

    # SVM
    svm_tfidf = SVC(random_state= 1, C = 10, gamma = 10, kernel='sigmoid', decision_function_shape='ovo').fit(x_train_scale, y_train)
    # save model
    pickle.dump(svm_tfidf, open('../model/topic_classification/svm_topic_classification_basic.pkl', 'wb'))
    # predict test topic
    svm_y_predict = svm_tfidf.predict(x_test_scale) 
    # save classification report and confusion matrix in csv
    classification_df, confusion_matrix_df = save_result('./result/svm_result_base.csv', y_test, svm_y_predict)
    print(classification_df)
    print(confusion_matrix_df)
    # Accuracy:0.916437098

    # Hypertuning
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
    grid_result = grid.fit(x_train_scale,y_train)
    print(grid_result.best_params_)

    # Final SVC model
    svm_tfidf_final = SVC(random_state= 1, C = 10, gamma = 1, kernel='rbf', decision_function_shape='ovo').fit(x_train_scale, y_train)
    # save model
    pickle.dump(svm_tfidf_final, open('../model/topic_classification/svm_topic_classification_final.pkl', 'wb'))
    # predict test topic
    svm_y_predict_final = svm_tfidf_final.predict(x_test_scale) 
    # save classification report and confusion matrix in csv
    classification_df, confusion_matrix_df = save_result('./result/svm_result_final.csv', y_test, svm_y_predict_final)
    print(classification_df)
    print(confusion_matrix_df)
    # Accuracy: 0.977961433

    # FOR DEPLOYMENT PURPOSES
    train = processed_data
    y_train = processed_data['Dominant_Topic']
    train_data = train.Text.values.tolist()
    x_train_corpus = preprocess(train_data, dictionary)
    train_vecs = create_vectors(x_train_corpus, train, lda_tfidf_model, 3)
    # convert to numpy array 
    x_train = np.array(train_vecs)
    y_train = np.array(y_train)

    # Scale Data
    scaler = StandardScaler()
    x_train_scale = scaler.fit_transform(x_train)

    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(SVC(random_state=1),param_grid,refit=True,verbose=2)
    grid_result = grid.fit(x_train_scale,y_train)
    print(grid_result.best_params_)
   
    svm_tfidf_final = SVC(random_state= 1, C = 100, gamma = 1, kernel='rbf', decision_function_shape='ovo').fit(x_train_scale, y_train)
    # save model
    pickle.dump(svm_tfidf_final, open('../model/topic_classification/svm_topic_classification_deployment.pkl', 'wb'))
    

    
   

   
   
    