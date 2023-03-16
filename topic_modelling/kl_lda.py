import pandas as pd
import gensim
import gensim.corpora as corpora 
from pprint import pprint
import pyLDAvis.gensim
import pickle 
import pyLDAvis
import os
from pathlib import Path

# split a sentence into a list 
def split_sentence(text):
    lst = []
    for sentence in text:
        lst.append(sentence.split())
    return lst

if __name__ == '__main__':
    # read in post processed data
    processed_data = pd.read_csv('../data/curated/reviews/yiting_cleaned_reviews.csv')

    # combine all the processed text into a list
    data = processed_data.processed_text.values.tolist()

    # split sentences to individual words
    data_words = split_sentence(data)

    # Create dictionary 
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    collection_texts = data_words

    # Term Document Frequency
    # convert list of words into bag-of-words format
    corpus = [id2word.doc2bow(text) for text in collection_texts]

    # number of topics
    num_topics = 5

    # Build LDA model 
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)

    # Print the Keyword in the 10 topics
    #pprint(lda_model.print_topics())
    #doc_lda = lda_model[corpus]

    # Visualize the topics
    #pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('../model/kl_ldavis_tuned_'+str(num_topics)+'.pkl')
    filePath = Path(LDAvis_data_filepath)
    filePath.touch(exist_ok= True)
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    #load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, 'kl_ldavis_tuned_'+str(num_topics)+'.html')
    LDAvis_prepared
        