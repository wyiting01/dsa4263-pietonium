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

def read_data(path):
    """
    Read cleaned data
    """
    processed_data = pd.read_csv(path)
    return processed_data

def split_sentence(text):
    """
    Split a sentence into a list 
    """
    lst = []
    for sentence in text:
        lst.append(sentence.split())
    return lst

def preprocess_words(data_inv):
    """
    1. Split sentences to individual words
    2. Build bigram and trigram models
    3. Form bigram and trigram
    4. Create dictionary
    5. Create corpus
    """
    data_words = split_sentence(data_inv)

    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)  
    
    data_bigrams = make_bigrams(data_words, bigram_phrases)
    data_bigrams_trigrams = make_trigrams(data_bigrams, bigram_phrases, trigram_phrases)

    id2words = corpora.Dictionary(data_bigrams_trigrams)

    collection_texts = data_bigrams_trigrams
    bow_corpuss = [id2words.doc2bow(text) for text in collection_texts]

    return ([data_bigrams_trigrams, id2words, bow_corpuss])

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

def obtain_corpus(input_data):
    """
    Combine all processed text into a list and
    obtain preprocessed text, dictionary and bag of words corpus
    """
    data = input_data.processed_text.values.tolist()
    data_words, id2word, bow_corpus = preprocess_words(data)
    return (data_words, id2word, bow_corpus)

def create_tfidf_model(corpus):
    """
    Create tfidf corpus from bag of words corpus
    """
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf

def load_base_lda_model(data_words, id2word):
    """
    1. Load and return Base model 
    2. Print Baseline coherence score
    """
    lda_tfidf_model_baseline = gensim.models.LdaMulticore.load('../model/kl_lda_tfidf_model_baseline.pkl')
    coherence_model_lda = CoherenceModel(model=lda_tfidf_model_baseline, 
                                        texts = data_words, 
                                        dictionary=id2word,
                                        coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Baseline Coherence Score: ', coherence_lda)
    return lda_tfidf_model_baseline

def load_final_lda_model(data_words, id2word):
    """
    1. Load and return Final model 
    2. Print Final coherence score
    """    
    lda_tfidf_model = gensim.models.LdaMulticore.load('../model/kl_lda_tfidf_model.pkl')
    coherence_model_lda = CoherenceModel(model=lda_tfidf_model, 
                                        texts = data_words, 
                                        dictionary=id2word,
                                        coherence='c_v')
    print('Final Coherence Score:', coherence_model_lda)
    print('Final number of topics used:', number_of_topics(lda_tfidf_model))
    return lda_tfidf_model

def print_keywords_per_topic(lda_model, k):
    """
    Print k keywords for each topic 
    """
    for idx, topic in lda_model.print_topics(num_words=k):
        print('Topic: {} \nWords: {}'.format(idx, topic))

def number_of_topics(lda_model):
    """
    Return the optimal number of topics of a model
    """
    num = 0
    for idx, topic in lda_model.print_topics(num_words=10):
        num += 1
    return num

def visualize_topics(lda_model, corpus_tfidf, id2word):
    """
    Save the LDAvis file and load it to open the html of LDAvis
    """
    final_num_topics = number_of_topics(lda_model)
    LDAvis_data_filepath = os.path.join('../model/kl_ldavis_tfidf_'+str(final_num_topics)+'.pkl')
    filePath = Path(LDAvis_data_filepath)
    filePath.touch(exist_ok= True)
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus_tfidf, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, 'result/kl_ldavis_tfidf_'+str(final_num_topics)+'.html')
    LDAvis_prepared

# Finding the dominant topic for each review
def format_topics_sentences(chosen_model, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each review
    for i, row in enumerate(chosen_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = chosen_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return(sent_topics_df, df_dominant_topic)

def dominant_topic_per_review(lda_model, corpus_tfidf, data):
    """
    Find the dominant topic for each review and
    save it to csv for later use in topic classification
    """
    df_topic_sents_keywords, df_topic_per_key = format_topics_sentences(lda_model, corpus_tfidf, data)
    df_topic_per_key.to_csv('./result/dominant_topic_in_each_sentence.csv')
    return df_topic_sents_keywords

def topic_distri_across_doc(dorminant_topic_each_sent):
    """
    Count the number of reviews in each topic 
    """
    # Number of reviews for Each Topic
    topic_counts = dorminant_topic_each_sent['Dominant_Topic'].value_counts()

    # Percentage of reviews for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)

    # Topic Number and Keywords
    topic_num_keywords = dorminant_topic_each_sent[['Dominant_Topic', 'Topic_Keywords']]

    # Concatenate Column wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    return(df_dominant_topics)

def unique_keyword_per_topic(lda_model):
    """
    To find the unique sets of keywords in each topic
    """
    for idx, topic in lda_model.print_topics(num_words=10):    
        print('Topic: {} \nWords: {}'.format(idx, topic))
    my_dict = {'Topic_' + str(i): [token for token, score in lda_model.show_topic(i, topn=10)] for i in range(0, lda_model.num_topics)}
    topics_keywords = []
    for key,value in my_dict.items():
        topics_keywords.append(set(value))
    # find the intersection between the 3 topics
    result = topics_keywords[0].intersection(topics_keywords[1],topics_keywords[2])
    final_unique_set = []
    for i in range(len(topics_keywords)):
        for j in range(len(topics_keywords)):
            if j != i:
                for k in range(len(topics_keywords)):
                    if k not in [j,i]:
                        set_unique = (topics_keywords[i]^result^topics_keywords[j]^topics_keywords[k])&topics_keywords[i]
                        if set_unique not in final_unique_set:
                            final_unique_set.append(set_unique)
                        break
                break
    return (final_unique_set)

def unique_sets(lda_model):
    """
    Print out the unique sets of keywords in each topic
    """
    unique_sets = unique_keyword_per_topic(lda_model)
    for i in range (len(unique_sets)):
        print('Topic {}: {}'.format(i, unique_sets[i]))
