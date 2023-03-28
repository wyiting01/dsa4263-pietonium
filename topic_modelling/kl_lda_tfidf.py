# LDA model with Tfidf
# adapted from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# adapted from https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
# adapted from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
# adapted from https://github.com/wjbmattingly/topic_modeling_textbook/blob/main/03_03_lda_model_demo_bigrams_trigrams.ipynb
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
import matplotlib.pyplot as plt

# split a sentence into a list 
def split_sentence(text):
    lst = []
    for sentence in text:
        lst.append(sentence.split())
    return lst

def preprocess_words(data_inv):
    # split sentences to individual words
    data_words = split_sentence(data_inv)

    # Build the bigram and trigram models
    # bigrams are two words frequently occur together
    # trigrams are three words frequently occurring 
    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)  

    # form bigrams
    data_bigrams = make_bigrams(data_words, bigram_phrases)

    # form trigrams
    data_bigrams_trigrams = make_trigrams(data_bigrams, bigram_phrases, trigram_phrases)

    # Create dictionary 
    id2words = corpora.Dictionary(data_bigrams_trigrams)

    # Create Corpus
    collection_texts = data_bigrams_trigrams

    # Term Document Frequency
    # convert list of words into bag-of-words format
    bow_corpuss = [id2words.doc2bow(text) for text in collection_texts]

    return ([data_bigrams_trigrams, id2words, bow_corpuss])

def make_bigrams(texts, bigram_text):
    # faster way to detect bigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_text, trigram_text,):
    # faster way to detect bigrams/trigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    trigram_mod = gensim.models.phrases.Phraser(trigram_text)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# compute coherence values of different k
def compute_coherence_values(type_corpus, k):
    topics = []
    score = []
    flag = True
    optimal_topic_index = 0
    optimal_topic_no = 0

    # optimal number of topics start from 2 
    # one topic is "ignored"
    for i in range(2, k):
        lda_model = gensim.models.LdaMulticore(corpus=type_corpus,
                                               id2word=id2word,
                                                num_topics=i, 
                                                random_state=100,
                                                chunksize=100,
                                                passes=10)
        # instantiate topic coherence model
        cm = CoherenceModel(model=lda_model, texts= data_words, dictionary=id2word, coherence='c_v')
        # Append number of topics modeled
        topics.append(i)
        # Append coherence scores to list
        score.append(cm.get_coherence()) 
        # check for highest cm before flattening
        if i != 2 and flag == True:
            if cm.get_coherence() < score[i-3]:
                optimal_topic_no = i - 1
                optimal_topic_index = i - 3
                flag = False

    # print the coherence score of topic number from 2 to k
    for m, cv in zip(topics, score):  
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
    # save the coherence score and topic number as dataframe
    output = {'k': topics, 'coherent score': score}
    df_output = pd.DataFrame(output)
    
    # plot the graph of the coherence score and topic number 
    plt.plot(df_output['k'], df_output['coherent score'])
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.title("Coherence score with respective number of topics")
    plt.savefig('./result/coherence_plot.png', dpi=300, bbox_inches='tight')
    #plt.show()

    return [optimal_topic_no, score[optimal_topic_index], df_output]

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

# Find the most representative review for each topic
# purpose: topic keyword might not give much sense of what the topic is. Hence, find some reviews and infer the topic 
def most_representative_doc_per_topic(dorminant_topic_each_sent):
    # Group top 5 sentences under each topic
    sent_topics_sorteddf = pd.DataFrame()

    sent_topics_outdf_grpd = dorminant_topic_each_sent.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                            grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

    # Reset Index    
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    return(sent_topics_sorteddf)

# Topic distribution across reviews
def topic_distri_across_doc(dorminant_topic_each_sent):
    # Number of Documents for Each Topic
    topic_counts = dorminant_topic_each_sent['Dominant_Topic'].value_counts()

    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)

    # Topic Number and Keywords
    topic_num_keywords = dorminant_topic_each_sent[['Dominant_Topic', 'Topic_Keywords']]

    # Concatenate Column wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    return(df_dominant_topics)

if __name__ == '__main__':
    # read in post processed data
    processed_data = pd.read_csv('../data/curated/reviews/yiting_cleaned_reviews.csv')

    # combine all the processed text into a list
    data = processed_data.processed_text.values.tolist()

    # preprocess words
    data_words, id2word, bow_corpus = preprocess_words(data)

    # create tf-idf model
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
 
    # number of topics for baseline
    num_topics = 2

    # Running LDA using TF-IDF corpus
    lda_model_tfidf = gensim.models.LdaMulticore(corpus=corpus_tfidf, 
                                                 num_topics=num_topics, 
                                                 id2word=id2word, 
                                                 passes=2,
                                                 workers=4)

    # baseline model coherence
    coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts = data_words, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Baseline Coherence Score: ', coherence_lda)
    # Baseline Coherence Score: 0.3520876611629823

    # create directory to keep results
    os.makedirs('result/', exist_ok=True)

    # hyperparameter tuning on number of topics
    final_num_topics, final_score, coherence_score_topic = compute_coherence_values(corpus_tfidf,12)
    coherence_score_topic.to_csv('./result/coherence_score_topic.csv')
    print('Final Coherence Score:', final_score)
    print('Final number of topics used:', final_num_topics)
    # Final Coherence Score: 0.5535325140273738 
    # Final number of topics: 3
    
    # final model with parameters yielding highest coherence score
    final_lda_model_tfidf = gensim.models.LdaMulticore(corpus=corpus_tfidf,
                                        id2word=id2word,
                                        num_topics=final_num_topics,
                                        passes=2,
                                        workers=4)
    
    # Print the Keyword for each topic
    for idx, topic in final_lda_model_tfidf.print_topics(num_words=10):    
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # save model
    final_lda_model_tfidf.save('../model/kl_lda_tfidf_model.pkl')

    # Visualize the topics
    LDAvis_data_filepath = os.path.join('../model/kl_ldavis_tfidf_'+str(final_num_topics)+'.pkl')
    filePath = Path(LDAvis_data_filepath)
    filePath.touch(exist_ok= True)
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(final_lda_model_tfidf, corpus_tfidf, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    #load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, 'result/kl_ldavis_tfidf_'+str(final_num_topics)+'.html')
    LDAvis_prepared

    # load model
    lda_tfidf_model = gensim.models.LdaMulticore.load('../model/kl_lda_tfidf_model.pkl')
    
    # Find the dominant topic for each review
    df_topic_sents_keywords, df_topic_per_key = format_topics_sentences(lda_tfidf_model, corpus_tfidf, data)
    df_topic_per_key.to_csv('./result/dominant_topic_in_each_sentence.csv')

    # Find the most representative review for each topic
    contribution_per_topic = most_representative_doc_per_topic(df_topic_sents_keywords)
    contribution_per_topic.to_csv('./result/most_representative_document_for_each_topic.csv')

    # Topic distribution across documents
    df_dominant_topic = topic_distri_across_doc(df_topic_sents_keywords)
    df_dominant_topic.head(final_num_topics).to_csv('./result/topic_distribution_across_documents.csv')



    




