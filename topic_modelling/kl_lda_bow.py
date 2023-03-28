# LDA Model with BOW
import pandas as pd
import gensim
import gensim.corpora as corpora 
from gensim import models
import pyLDAvis.gensim
import pickle 
import pyLDAvis
import os
from pathlib import Path
from gensim.models import CoherenceModel
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
# split a sentence into a list 
# remove words with length lesser than 3
def split_sentence(text):
    lst = []
    for sentence in text:
        splitted_sentence = sentence.split()
        inner_lst = []
        for word in splitted_sentence:
            if len(word) >= 3:
                inner_lst.append(word)
        lst.append(inner_lst)
    return lst
"""

# split a sentence into a list 
def split_sentence(text):
    lst = []
    for sentence in text:
        lst.append(sentence.split())
    return lst

# compute coherence values of different k
def compute_coherence_values(type_corpus, k):
    topics = []
    score = []
    for i in range(1, k):
        lda_model = gensim.models.LdaMulticore(corpus=type_corpus,
                                               id2word=id2word,
                                                num_topics=k, 
                                                random_state=100,
                                                chunksize=100,
                                                passes=10)
        # instantiate topic coherence model
        cm = CoherenceModel(model=lda_model, texts= data_words, dictionary=id2word, coherence='c_v')
        # Append number of topics modeled
        topics.append(i)
        # Append coherence scores to list
        score.append(cm.get_coherence()) 
        """
        # contain coherence scores of corresponding number of topics
        output = {'k': topics, 'coherent score': score}
        df_output = pd.DataFrame(output)
        """
        # get the index of highest coherence score
        array_score = np.array(score)
        max_index = array_score.argmax()
    # return the number of topics with the highest coherence score
    return [topics[max_index], max(score)]

def preprocess_words(data_inv):
    # split sentences to individual words
    data_word = split_sentence(data_inv)

    # Create dictionary 
    id2words = corpora.Dictionary(data_word)

    # Create Corpus
    collection_texts = data_word

    # Term Document Frequency
    # convert list of words into bag-of-words format
    bow_corpuss = [id2words.doc2bow(text) for text in collection_texts]

    return ([data_word, id2words, bow_corpuss])

if __name__ == '__main__':
    # read in post processed data
    processed_data = pd.read_csv('../data/curated/reviews/yiting_cleaned_reviews.csv')

    # combine all the processed text into a list
    data = processed_data.processed_text.values.tolist()

    # split test-train in ratio 20:80
    # included random state parameter so that results are reproducible
    training_data, testing_data = train_test_split(data, test_size =0.2, random_state=25)

    # preprocess words
    data_words, id2word, bow_corpus = preprocess_words(training_data)
 
    # number of topics
    num_topics = 5

    # Running LDA using Bag of Words for training data
    lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)

    # Print the Keyword in the 5 topics 
    for idx, topic in lda_model.print_topics(num_words=10):    
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # baseline model coherence
    coherence_model_lda = CoherenceModel(model=lda_model, texts = data_words, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Baseline Coherence Score: ', coherence_lda)

    # hyperparameter tuning on number of topics
    final_num_topics, final_score = compute_coherence_values(bow_corpus,12)
    print('Final Coherence Score:', final_score)
    print('Final number of topics used:', final_num_topics)

    # final model with parameters yielding highest coherence score
    final_lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                        id2word=id2word,
                                        num_topics=final_num_topics)
    # save trained model
    final_lda_model.save('../model/kl_lda_bow_model.pkl')
   
    # Print the Keyword in optimal model
    for idx, topic in final_lda_model.print_topics(num_words=10):    
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # Use test data on trained LDA_model
    # split test data sentences to individual words
    data_word_test = split_sentence(testing_data)

    # Create Corpus for test data using trained dictionary
    collection_text_test = data_word_test

    # convert list of words into bag-of-words format
    bow_corpus_test = [id2word.doc2bow(text) for text in collection_text_test]

    # load model 
    lda_model_test = gensim.models.LdaMulticore.load('../model/kl_lda_bow_model.pkl')
    #trial1 = lda_model_test.get_document_topics(bow_corpus_test)
    a = []
    for query in bow_corpus_test:
        for i in lda_model_test[query]:
            a.append(i)
    b = list(sorted(a, key=lambda x: x[1]))
    b.reverse()
    print(b[:10])
    for j in range(final_num_topics):
        print(lda_model_test.print_topic(b[j][0]), '\n')   

    """
    # Visualize the topics
    LDAvis_data_filepath = os.path.join('../model/kl_ldavis_bow_'+str(final_num_topics)+'.pkl')
    filePath = Path(LDAvis_data_filepath)
    filePath.touch(exist_ok= True)
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(final_lda_model, bow_corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    #load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, 'kl_ldavis_bow_'+str(final_num_topics)+'.html')
    """

    

    
        


    




