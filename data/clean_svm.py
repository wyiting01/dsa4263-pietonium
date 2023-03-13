from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import WordNetLemmatizer
import re
import nltk
import pandas as pd
import os

# nltk.download('stopwords')

def clean_text1(text):
    """
    This function takes as input a text on which several 
    NLTK algorithms will be applied in order to preprocess it
    """
    tokens = word_tokenize(text)
    # Remove the punctuations
    tokens = [word for word in tokens if word.isalpha()]
    # Lower the tokens
    tokens = [word.lower() for word in tokens]
    # Remove stopword
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    # Lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word, pos = "v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos = "n") for word in tokens]
    return tokens

# POS Tags to be kept (Noun, Verb, Adjective, Adverb) (n,v,a,r)
KEPT_POSTAGS = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VBZ', 'VBP', 'VBN', 'VBG','VBD', 'VB', 'RBS', 'RB', 'RBR']
NOUN_POSTAGS = ['NN', 'NNS', 'NNP', 'NNPS']
VERB_POSTAGS = ['VBZ', 'VBP', 'VBN', 'VBG','VBD', 'VB']

def mylemmatize(word, pos):
    lemma = WordNetLemmatizer()
    if pos in VERB_POSTAGS:
        return lemma.lemmatize(word, pos = 'v')
    elif pos in NOUN_POSTAGS:
        return lemma.lemmatize(word, pos = 'n')
    else:
        return word
    
def clean_text2(text):
    tokens = word_tokenize(text)
    # Only keep text
    tokens = [word for word in tokens if word.isalpha()]
    # Lower the tokens
    tokens = [word.lower() for word in tokens]
    # Remove stopword
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    # only keep noun, verb, adverb, adjective
    tokens = [mylemmatize(*word_tup) for word_tup in nltk.pos_tag(tokens)]

    return tokens

HTML_PATTERN = re.compile('<.*?>')
def noise_entity_removal(target_input):
    # convert to lower case
    target_input = target_input.lower()
    
    # remove html tags
    target_input = re.sub(HTML_PATTERN, '', target_input)
    
    # remove non-word characters like #,*,% etc
    target_input = re.sub(r'\W',' ', target_input)
    
    #will remove digits
    target_input = re.sub(r'\d',' ',target_input)
    
    #will remove extra spaces
    target_input = re.sub(r'\s+',' ',target_input)
    
    # remove stopwords
    target_input_tokens = nltk.word_tokenize(target_input)
    target_input_tokens_wo_stopwords = [i for i in target_input_tokens if i not in stopwords.words('english') and i]
    
    # join the list of tokens back to string
    output = " ".join(target_input_tokens_wo_stopwords)
    
    return output

def join_tokens(token_list):
    """
    Joins tokens in the list into a complete string
    """
    return " ".join(token_list)


# DATA PRE-PROCESS FOR CURATED LAYER SVM
data = pd.read_csv('./raw/reviews.csv')
data.head()

def clean_method_1(text):
    return join_tokens(clean_text2(noise_entity_removal(text)))

data.Text = data.Text.apply(lambda x: clean_method_1(x))
data.Sentiment = data.Sentiment.apply(lambda x: 1 if str.lower(x) == 'positive' else 0)
data = data[['Text', 'Sentiment']]
os.makedirs('./curated/reviews/svm', exist_ok=True)
data.to_csv('./curated/reviews/svm/cleaned1.csv', index=False)
