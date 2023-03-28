# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Noise Entity Removal
'''
This section is about noise entity removal.
First, we convert all the words to lower case. We then remove html tags, non-word characters, digits and extra spaces.
Finally, we remove stopwords from the documents.
'''
HTML_PATTERN = re.compile('<.*?>')
nltk.download('stopwords')
STOPWORDS_LIST = set(stopwords.words('english'))

def noise_entity_removal(target_input):
    # convert to lower case
    target_input = target_input.lower()
    
    # remove html tags
    target_input = re.sub(HTML_PATTERN, ' ', target_input)
    
    # remove non-word characters like #,*,% etc
    target_input = re.sub(r'\W',' ', target_input)
    
    #will remove digits
    target_input = re.sub(r'\d',' ',target_input)
    
    #will remove extra spaces
    target_input = re.sub(r'\s+',' ',target_input)
    
    # remove stopwords
    target_input_tokens = nltk.word_tokenize(target_input)
    target_input_tokens_wo_stopwords = [i for i in target_input_tokens if i not in STOPWORDS_LIST and i]
    
    # join the list of tokens back to string
    output = " ".join(target_input_tokens_wo_stopwords)
    
    return output

# Text Normalization
'''
In this section, we normalize our documents by either stemming or lemmatizing.
Since lemmatization is able to retain the sentiment meanings, we will make lemmatization as the default.
We are also removing tokens with length less than 3.
'''
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

# POS Tags to be kept (Noun, Verb, Adjective, Adverb) (n,v,a,r)
KEPT_POSTAGS = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VBZ', 'VBP', 'VBN', 'VBG','VBD', 'VB', 'RBS', 'RB', 'RBR']
NOUN_POSTAGS = ['NN', 'NNS', 'NNP', 'NNPS']
VERB_POSTAGS = ['VBZ', 'VBP', 'VBN', 'VBG','VBD', 'VB']

def mylemmatize(word, pos):
    if pos in VERB_POSTAGS:
        return LEMMATIZER.lemmatize(word, pos = 'v')
    elif pos in NOUN_POSTAGS:
        return LEMMATIZER.lemmatize(word, pos = 'n')
    else:
        return word

def text_normalization(target_input, method = 'lemmatize'):
    target_input_tokens = nltk.word_tokenize(target_input)
    
    if method == 'lemmatize':
        lemmatized_tokens = [mylemmatize(*word_tup) for word_tup in nltk.pos_tag(target_input_tokens)]
        revised_lemmatized_tokens = [i for i in lemmatized_tokens if len(i) >= 3]
        output = " ".join(revised_lemmatized_tokens)
    
    if method == 'stem':
        stemmed_tokens = [STEMMER.stem(word) for word in target_input_tokens]
        revised_stemmed_tokens = [i for i in stemmed_tokens if len(i) >= 3]
        output = " ".join(revised_stemmed_tokens)
        
    return output

# Standardise Labels
'''
Since the dataset labels for sentiments are in the form of words (i.e. positive and negative), we will convert these labels to integers instead.
Positive: 1
Negative: 0
'''
def label_to_integer(sentiment_label):
    if sentiment_label == 'positive':
        return 1
    elif sentiment_label == 'negative':
        return 0
    else:
        return None
    
# Start Preprocessing Steps on the Raw Reviews Dataset
print('')
df = pd.read_csv('raw/reviews.csv')
print('Preprocessing reviews.csv...')
df['processed_text'] = df['Text'].apply(lambda x:noise_entity_removal(x))
df['processed_text'] = df['processed_text'].apply(lambda x:text_normalization(x))
df['Sentiment'] = df['Sentiment'].apply(lambda x:label_to_integer(x))
print('Preprocessing Completed!')

# Save Processed Dataset
df.to_csv('curated/reviews/yiting_cleaned_reviews.csv', index = False)
print('Saved as CSV :)')
