# RUN pip install -r DistilBERT1_requirements.txt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re 

from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
import datasets
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import logging


## OVERALL FUNCTIONS THAT CAN BE USED BOTH TRAINING AND INFERENCING ##
CONFIG = {
    'this_time': 1,
    'random_seed': 4263,
    'test_size': 0.2,
    'val_size': 0.25, # relative to the remaining training data -> (train, val, test) = (60, 20, 20),
    'pretrained_model': 'distilbert-base-uncased',
    'learning_rate': 2e-5,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 16,
    'num_train_epochs': 10,
    'weight_decay': 0.01,
    'evaluation_strategy' : "epoch",
    'save_strategy': "epoch",
    'output_dir': "DistilBERT1"
}
HTML_PATTERN = re.compile('<.*?>')

def read_data(path):
    """
    Read cleaned text data with columns 'Sentiment' and 'processed_text'
    and return the result pd.DataFrame

    :param path: path to the csv data
    :type path: str

    Output: pd.DataFrame
    """
    cleaned_df = pd.read_csv(path)
    return cleaned_df

def make_dataset(df):
    """
    Read the cleaned df and return the Dataset object that is fed to model
    """
    dataset = datasets.Dataset.from_dict({
        'text': df['processed_text'],
        'label': df['Sentiment']
    })
    return dataset


## FUNCTIONS FOR EVALUATION (notebook) AND SCORING (app) ##

def load_distilbert_model():
    """
    Load from Hugging Face and return a tuple of (model, tokenizer) 
    """
    try:
        model_link = "DreamyBeaver/distilBERT-SA-pietonium"
        model = AutoModelForSequenceClassification.from_pretrained(model_link)
        tokenizer = AutoTokenizer.from_pretrained(model_link) 

        return model, tokenizer
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function load_disilbert_model')
        error_message = str(exc.replace("\n", ""))
        return error_message


def noise_remove_helper(raw_text):
    """
    Helper for both single scoring and file scoring
    Basic noise removal in the raw text (no stopwords removal, stemming or lemmatizing)

    :param raw_text: raw review user put in
    :type raw_text: str
    """

    #will remove digits
    target_input = re.sub(r'\d',' ',raw_text)
    
    # convert to lower case
    target_input = target_input.lower()
    
    # remove html tags
    target_input = re.sub(HTML_PATTERN, ' ', target_input)
    
    # remove non-word characters like #,*,% etc
    target_input = re.sub(r'\W',' ', target_input)
    
    #remove words less than 3 characters
    target_input = re.sub(r'\b\w{2}\b', '', target_input)

    ##will remove extra spaces
    target_input = re.sub(r'\s+',' ',target_input)

    return target_input 

def label_to_integer_helper(label):
    if 'negative' in str.lower(label):
        return 0
    elif 'positive' in str.lower(label):
        return 1
    else:
        return None

# For App
def scoring_single_review(raw_review, model, tokenizer):
    """
    Read a raw review and return tuple of (predicted sentiment, predicted sentiment probability)
    """
    try:
        cleaned_review = noise_remove_helper(raw_review)
        
        input = tokenizer(cleaned_review, return_tensors='pt', truncation=True)
        with torch.no_grad():
            logits = model(**input).logits
        predicted_class_id = logits.argmax().item()
        predicted_class_prob = round(F.softmax(logits, dim=1).flatten()[predicted_class_id].item(), 5)
            
        return predicted_class_id, predicted_class_prob
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function scoring_single_review')
        error_message = str(exc.replace("\n", ""))
        return error_message

def id_2_label(model, id):
    """
    Return the text label Negative / Positive from predicted labels 0 / 1
    """
    return model.config.id2label[id]

def prepare_inference_data(filename):
    """
    filename: reviews_test.csv
    schema: Text | Time
    
    Output: pd.DataFrame of columns Text | Time | processed_text
    """
    try:
        df = pd.read_csv(filename)
        df = df.loc[~df['Text'].isna()]
        df['processed_text'] = df['Text'].apply(lambda x: noise_remove_helper(x))
        return df
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function prepare_inference_data')
        error_message = str(exc.replace("\n", ""))
        return error_message


# For App and Presentation
def scoring_file_thread(filename, model, tokenizer): 
    """
    Read the filename and return the pd.DataFrame of columns ['Time', 'Text', 'predicted_sentiment', 'predicted_sentiment_probability']
    """
    try:
        df = prepare_inference_data(filename)
        reviews = np.array(df['processed_text'])

        with ThreadPoolExecutor(max_workers=500) as executor:
            future_result_list = [executor.submit(scoring_single_review, review, model, tokenizer) for review in reviews]
        result_list = [x.result() for x in concurrent.futures.as_completed(future_result_list)]
        final_df = pd.concat([df[['Time', 'Text']], pd.DataFrame(data=result_list, columns=['predicted_sentiment', 'predicted_sentiment_probability'])], axis=1)
        final_df['predicted_sentiment'] = final_df['predicted_sentiment'].apply(lambda x: id_2_label(model,x))
        return final_df
    
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function scoring_file_thread')
        error_message = str(exc.replace("\n", ""))
        return error_message

def scoring_file_thread_df(df, model, tokenizer): 
    """
    Read a dataframe (pd.DataFrame) and return the pd.DataFrame of columns ['Time', 'Text', 'predicted_sentiment', 'predicted_sentiment_probability']
    """
    try:
        df = df.loc[~df['Text'].isna()]
        df['processed_text'] = df['Text'].apply(lambda x: noise_remove_helper(x))
        reviews = np.array(df['processed_text'])

        with ThreadPoolExecutor(max_workers=500) as executor:
            future_result_list = [executor.submit(scoring_single_review, review, model, tokenizer) for review in reviews]
        result_list = [x.result() for x in concurrent.futures.as_completed(future_result_list)]
        final_df = pd.concat([df[['Time', 'Text']], pd.DataFrame(data=result_list, columns=['predicted_sentiment', 'predicted_sentiment_probability'])], axis=1)
        final_df['predicted_sentiment'] = final_df['predicted_sentiment'].apply(lambda x: id_2_label(model,x))
        return final_df
    
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function scoring_file_thread_df')
        error_message = str(exc.replace("\n", ""))
        return error_message


# For App and Presentation
def scoring_file_dummy(filename, model, tokenizer):
    """
    Read the filename and return the pd.DataFrame of columns []'Time', 'Text', 'predicted_sentiment', 'predicted_sentiment_probability']

    """
    try:
        df = pd.read_csv(filename)
        df = df.loc[~df['Text'].isna()]
        df['processed_text'] = df['Text'].apply(lambda x: noise_remove_helper(x))

        predicted_labels = []
        predicted_label_probs = []
        with torch.no_grad():
            for text in df['processed_text']:
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                logits = model(**inputs).logits
                predicted_class_id = logits.argmax().item()
                predicted_class_prob = round(F.softmax(logits, dim=1).flatten()[predicted_class_id].item(), 3)
                predicted_labels.append(predicted_class_id)
                predicted_label_probs.append(predicted_class_prob)
        df['predicted_sentiment'] = pd.Series(predicted_labels)
        df['predicted_sentiment'] = df['predicted_sentiment'].apply(lambda x: id_2_label(model,x))
        df['predicted_sentiment_probability'] = pd.Series(predicted_label_probs)
        return df[['Time', 'Text', 'predicted_sentiment', 'predicted_sentiment_probability']]
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function scoring_file_dummy')
        error_message = str(exc.replace("\n", ""))
        return error_message

# For App and Presentation
def scoring_file_dummy_df(df, model, tokenizer):
    """
    Read the dataframe (pd.DataFrame) and return the pd.DataFrame of columns []'Time', 'Text', 'predicted_sentiment', 'predicted_sentiment_probability']

    """
    try:
        df = df.loc[~df['Text'].isna()]
        df['processed_text'] = df['Text'].apply(lambda x: noise_remove_helper(x))

        predicted_labels = []
        predicted_label_probs = []
        with torch.no_grad():
            for text in df['processed_text']:
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                logits = model(**inputs).logits
                predicted_class_id = logits.argmax().item()
                predicted_class_prob = round(F.softmax(logits, dim=1).flatten()[predicted_class_id].item(), 3)
                predicted_labels.append(predicted_class_id)
                predicted_label_probs.append(predicted_class_prob)
        df['predicted_sentiment'] = pd.Series(predicted_labels)
        df['predicted_sentiment'] = df['predicted_sentiment'].apply(lambda x: id_2_label(model,x))
        df['predicted_sentiment_probability'] = pd.Series(predicted_label_probs)
        return df[['Time', 'Text', 'predicted_sentiment', 'predicted_sentiment_probability']]
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function scoring_file_dummy_df')
        error_message = str(exc.replace("\n", ""))
        return error_message


# For evaluate.py (and App if Sentiment(label) is provided)
def visualize_confusion_matrix(predicted_labels, real_labels):
    """
    :param predicted_labels: This is an array-like with values that are 0 or 1, shape (n_samples,)
    :param real_labels: This is an array with values that are 0 or 1, shape (n_samples,)
    :return: show figure
    """
    try:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        import matplotlib.pyplot as plt

        cm = confusion_matrix(predicted_labels, real_labels)

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)

        plt.show()
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function visualize_confusion_matrix')
        error_message = str(exc.replace("\n", ""))
        return error_message

# For evaluate.py (and App if Sentiment(label) is provided)
def confusion_matrix(predicted_labels, real_labels):
    try:
        from sklearn.metrics import classification_report
        report = classification_report(real_labels, predicted_labels, output_dict=True)
        return report
    # print('positive: ', report['1'])
    # print('negative: ', report['0'])
    # print('accuracy: ', report['accuracy'])
    # For evaluate.py
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function confusion_matrix')
        error_message = str(exc.replace("\n", ""))
        return error_message

# for evaluate.py
def predict(input, model, tokenizer):
    """
    :param input: processed input (ex: processed_text) of array-like with shape (n,) OR single review (string)

    Output:
        If single review (str) -> return (predicted_class_id, predicted_class_prob) by function scoring_single_review
        
        If array-like -> return dict of {'preds': predicted_labels, 'preds_prob': predicted_label_probs}
    """
    try:
        if type(input) == str:
            return scoring_single_review(input, model, tokenizer)
        else:
            predicted_labels = []
            predicted_label_probs = []
            with torch.no_grad():
                for text in input:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True)
                    logits = model(**inputs).logits
                    predicted_class_id = logits.argmax().item()
                    predicted_class_prob = round(F.softmax(logits, dim=1).flatten()[predicted_class_id].item(), 3)
                    predicted_labels.append(predicted_class_id)
                    predicted_label_probs.append(predicted_class_prob)
            return {'preds': predicted_labels, 'preds_prob': predicted_label_probs}
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function predict()')
        error_message = str(exc.replace("\n", ""))
        return error_message