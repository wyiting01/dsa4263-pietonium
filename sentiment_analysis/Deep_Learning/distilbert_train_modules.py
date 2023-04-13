## FUNCIONS ONLY FOR TRAINING ##
# These functions below are modules used in training pipeline #
import torch
import torch.nn.functional as F
import datasets
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import pandas as pd
import re
import logging

from distilbert import label_to_integer_helper, noise_remove_helper

# HELPER FUNCTIONS 
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

def visualize_confusion_matrix(predicted_labels, real_labels):
    """
    :param predicted_labels: This is an array-like with values that are 0 or 1, shape (n_samples,)
    :param real_labels: This is an array with values that are 0 or 1, shape (n_samples,)
    :return:
    """

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    import matplotlib.pyplot as plt

    cm = confusion_matrix(predicted_labels, real_labels)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)

    plt.show()

## FUNCIONS ONLY FOR TRAINING ##
# These functions below are modules used in training pipeline #
def build_config(config):
    """
    Return model architecture, training arguments, training tokenizer, data collator

    :param config: configuration of model hyperparameters and variables used in training
    :type config: dict
    """
    id2label = {0: 'Negative', 1: 'Positive'}
    label2id = {'Negative': 0, 'Positive': 1} 

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config['pretrained_model'], num_labels=2, id2label=id2label, label2id=label2id
        )
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            learning_rate=config['learning_rate'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            per_device_eval_batch_size=config['per_device_eval_batch_size'],
            num_train_epochs=config['num_train_epochs'],
            weight_decay=config['weight_decay'],
            evaluation_strategy=config['evaluation_strategy'],
            save_strategy=config['save_strategy'],
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        training_tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])
        data_collator = DataCollatorWithPadding(tokenizer=training_tokenizer)
        
        return model, training_args, training_tokenizer, data_collator

    except Exception as e:
        import traceback 
        exc = traceback.format_exc()
        error_message = str(exc.replace("\n", ""))
        return error_message
    
def preprocess_data_for_train(filename):
    try:
        df = pd.read_csv(filename)
        df = df.loc[~df['Text'].isna()]
        df['processed_text'] = df['Text'].apply(lambda x: noise_remove_helper(x))
        df['Sentiment'] = df['Sentiment'].apply(lambda x: label_to_integer_helper(x))
        return df[['Sentiment', 'processed_text']]
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        logging.error('Error from function prepare_inference_data')
        error_message = str(exc.replace("\n", ""))
        return error_message
    
def split_data(df, config):
    """
    Split the dataframe into training, evaluating, and testing data

    :param df: input dataframe
    :type df: pd.DataFrame
    :param config: configuration of model hyperparameters and variables used in training
    :type config: dict

    Output: tuple of (train df, eval df, test df)
    Format: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    try:
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(df, test_size = config['test_size'], random_state=config['random_seed'], shuffle=True)
        train_df, val_df = train_test_split(train_df, test_size = config['val_size'], random_state=config['random_seed'], shuffle=True)
        
        return (train_df, val_df, test_df)
    except Exception as e:
        import traceback 
        exc = traceback.format_exc()
        error_message = str(exc.replace("\n", ""))
        return error_message

def prepare_training_data(df, config, training_tokenizer):
    """
    create dataset objects of training, validating and testing data, full dataset and tokenized dataset

    :param df: available data for pipeline (which will be splitted into train, val, test)
    :type df: pd.DataFrame
    :param config: configuration of model hyperparameters and variables used in training
    :type config: dict
    :param training_tokenizer: returned from build_config()
    :type training_tokenizer: transformers.AutoTokenizer
    """
    try:
        train_df, val_df, test_df = split_data(df, config=config)

        train_dataset = make_dataset(train_df)
        val_dataset = make_dataset(val_df)
        test_dataset = make_dataset(test_df)

        full_dataset = datasets.DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'val': val_dataset
        })

        
        def preprocess_function(examples):
            return training_tokenizer(examples['text'], truncation=True)

        tokenized_dataset = full_dataset.map(preprocess_function, batched=True)
        return tokenized_dataset, full_dataset, train_dataset, val_dataset, test_dataset
    
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        error_message = str(exc.replace("\n", ""))
        return error_message


accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """
    the metrics will be passed to model configuration as assessment for training
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)



def train(model, training_args, tokenized_dataset, tokenizer, data_collator, compute_metrics):
    """
    Run the training process

    :params model, training_args, tokenizer, data_collator: return from build_config(**argss)
    :param tokenized_dataset: returned from prepare_training_data(**args)
    :param compute_metrics: a function to be passed into trainer 
    """
    try:
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        error_message = str(exc.replace("\n", ""))
        return error_message


def load_model_dynamic(model_link):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model = AutoModelForSequenceClassification.from_pretrained(model_link)
        tokenizer = AutoTokenizer.from_pretrained(model_link)

        return model, tokenizer
    except Exception as e:
        import traceback
        exc = traceback.format_exc()
        error_message = str(exc.replace("\n", ""))
        return error_message

def hasGPU():
    import torch
    torch.cuda.is_available()