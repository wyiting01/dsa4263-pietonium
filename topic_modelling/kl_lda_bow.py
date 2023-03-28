# LDA Model with BOW （Baseline model)
# adapted from https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
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

# split a sentence into a list 
def split_sentence(text):
    lst = []
    for sentence in text:
        lst.append(sentence.split())
    return lst

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

    # preprocess words
    data_words, id2word, bow_corpus = preprocess_words(data)

    # number of topics
    num_topics = 5

    # Running LDA using Bag of Words for data
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
    # Baseline cohorence scorce: 0.39132961449348624

    # save model
    lda_model.save('../model/kl_lda_bow_model.pkl')

    # create directory to keep results
    os.makedirs('result/', exist_ok=True)
 
    # Visualize the topics
    LDAvis_data_filepath = os.path.join('../model/kl_ldavis_bow_'+str(num_topics)+'.pkl')
    filePath = Path(LDAvis_data_filepath)
    filePath.touch(exist_ok= True)
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, bow_corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    #load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, 'result/kl_ldavis_bow_'+str(num_topics)+'.html')
    


    
        


    



