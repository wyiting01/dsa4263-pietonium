import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import ssl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Prevent error during nltk.download('stopwords')
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

HTML_PATTERN = re.compile('<.*?>')
STOPWORDS_LIST = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

# POS Tags to be kept (Noun, Verb, Adjective, Adverb) (n,v,a,r)
KEPT_POSTAGS = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VBZ', 'VBP', 'VBN', 'VBG','VBD', 'VB', 'RBS', 'RB', 'RBR']
NOUN_POSTAGS = ['NN', 'NNS', 'NNP', 'NNPS']
VERB_POSTAGS = ['VBZ', 'VBP', 'VBN', 'VBG','VBD', 'VB']

def noise_entity_removal(target_input):
    """
    Noise entity removal - First, we convert all the words to lower case. 
    We then remove html tags, non-word characters, digits and extra spaces. 
    Finally, we remove stopwords from the documents.
    :return: string
    """
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

def mylemmatize(word, pos):
    """Function to lemmatize input word
    :return: str
    """
    if pos in VERB_POSTAGS:
        return LEMMATIZER.lemmatize(word, pos = 'v')
    elif pos in NOUN_POSTAGS:
        return LEMMATIZER.lemmatize(word, pos = 'n')
    else:
        return word

def text_normalization(target_input, method = 'lemmatize'):
    """Normalise documents by lemmatizing
    :return: str
    """
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

def label_to_integer(sentiment_label):
    """Conversion of str labels to integer representation
        :return: int
    """
    if sentiment_label == 'positive':
        return 1
    elif sentiment_label == 'negative':
        return 0
    else:
        return None
    
def integer_to_label(sentiment_int):
    if sentiment_int == 1:
        return 'positive'
    elif sentiment_int == 0:
        return 'negative'
    else:
        return None
    
def preprocess(dataset, text_col_name = 'Text', label_col_name = None):
    # process the text column
    df = dataset.copy()
    df['processed_text'] = df[text_col_name].apply(lambda x:noise_entity_removal(x))
    df['processed_text'] = df['processed_text'].apply(lambda x:text_normalization(x))

    # process the label column if present and if its not in int64 type
    if label_col_name and label_col_name in df.columns:
        if df[label_col_name].dtypes != 'int64':
            df[label_col_name] = df[label_col_name].apply(lambda x:label_to_integer(x))
        
    return df

def split_train_test(features, labels):
    return train_test_split(features, labels, test_size = 0.2, random_state=4263)