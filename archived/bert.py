import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report

import tensorflow as tf

from transformers import BertTokenizer
from transformers import BertConfig
from transformers import TFBertForSequenceClassification

MAX_LEN = 512

def read_data(path):
    """Read cleaned text data with columns 'Sentiment' and 'processed_text'
        and rename the columns
    """
    cleaned_df = pd.read_csv(path)
    # Re-labelling of columns headers
    cleaned_df.rename(columns = {'Sentiment' : 'labels', 'processed_text' : 'text'}, inplace = True)

    # Extracting out the necessary columns
    cleaned_df = cleaned_df[['text','labels']]
    return cleaned_df

def labels_array(df):
    # Extract labels values to get the size
    arr = df['labels'].values
    
    # Creating 2D array to indicate which row of data the label belongs to
    labels = np.zeros((arr.size, arr.max() + 1), dtype=int)
    print(f"Label Shape: {labels.shape}")

    # Indicating the label (0 or 1) of the respective row of data
    ## [1, 0] indicates negative sentiment
    ## [0, 1] indicates positive sentiment
    labels[np.arange(arr.size), arr] = 1
    print(f"Label Shape: {labels.shape}")
    return labels

def tokenization(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    Xids = np.zeros((len(df), MAX_LEN))
    Xmask = np.zeros((len(df), MAX_LEN))
    for i, sequence in enumerate(df['text']):
        # Return a dictionary containing the encoded sentence
        tokens = tokenizer.encode_plus(str(sequence), max_length = MAX_LEN, 
                                   truncation = True,               # Needed since there are text seq > 512
                                   padding = "max_length",          # For sentence < 512, padding is applied to reach a length of 512
                                   add_special_tokens = True,       # Mark the start and end of sequences
                                   return_token_type_ids = False, 
                                   return_attention_mask = True, 
                                   return_tensors = 'tf')           # Return TensorFlow object
        # Retrieve input_ids and attention_mask
        ### input_ids : list of integers uniquely tied to a specific word
        ### attention_mask : binary tokens indicating which tokens are the actual input tokens and which are padding tokens
        Xids[i, :], Xmask[i, :] = tokens['input_ids'], tokens['attention_mask']
    return (Xids, Xmask)

def dataset(df):
    '''
    Format inputs to model 
    '''
    Xids, Xmask = tokenization(df)
    labels = labels_array(df)

    # Combine arrays into tensorflow object
    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

    def map_func(input_ids, masks, labels):
        """Function to restructure the dataset
        """
        return {'input_ids': input_ids, 'attention_mask': masks}, labels

    dataset = dataset.map(map_func)
    
    # Shuffle the data to prevent overfitting
    dataset = dataset.shuffle(100000, reshuffle_each_iteration=False)

    return dataset

def split_dataset(dataset):
    DS_LEN = len(list(dataset))
    SPLIT = .8
    # take or skip the specified number of batches to split by factor
    test = dataset.skip(round(DS_LEN * SPLIT)).batch(32)
    trainevalu = dataset.take(round(DS_LEN * SPLIT)) 
    DS_LEN2 = len(list(trainevalu))

    train = trainevalu.take(
            round(DS_LEN2 * SPLIT)).shuffle(1000, reshuffle_each_iteration=False).batch(32)
    evalu = trainevalu.skip(
            round(DS_LEN2 * SPLIT)).shuffle(1000, reshuffle_each_iteration=False).batch(32)
    return (test, train, evalu)

def build_config():
    bertConfig = BertConfig.from_pretrained('bert-base-uncased', 
                                        output_hidden_states = True,
                                        num_labels = 2,
                                        max_length = MAX_LEN
                                        )
    tranformersPreTrainedModelName = 'bert-base-uncased'
    bert = TFBertForSequenceClassification.from_pretrained(tranformersPreTrainedModelName, config = bertConfig)
    return bert


def build_layers():
    # Build 2 input layers to Bert Model where name needs to match the input values in the dataset
    input_ids = tf.keras.Input(shape = (MAX_LEN, ), name = 'input_ids', dtype = 'int32')
    mask = tf.keras.Input(shape = (MAX_LEN, ), name = 'attention_mask', dtype = 'int32')

    bert = build_config()
    # Consume the last_hidden_state from BERT
    embedings = bert.layers[0](input_ids, attention_mask=mask)[0]

    # Original Author: Ferry Djaja
    # https://djajafer.medium.com/multi-class-text-classification-with-keras-and-lstm-4c5525bef592
    X = tf.keras.layers.Flatten()(embedings)
    X = tf.keras.layers.Dropout(0.5)(X)
    y = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(X)

    model = tf.keras.Model(inputs=[input_ids,mask], outputs=y)
    #model.summary()
    return model

def load_model():
    pass

def evaluate_model(model,test_data):
    """Function to evaluate model and output confusion matrix with accuracy score
    Test data input 
    """
    # Run the model on the test data
    predicted = model.predict(test_data)

    y_pred = tf.nn.softmax(predicted)
    y_pred_argmax = tf.math.argmax(y_pred, axis=1)

    # unbatching the test data to obtain the labels
    test_un = test_data.unbatch()
    lst = []

    for features, label in test_un.take(-1):
        lab = tf.math.argmax(label).numpy()
        lst.append(lab)
        
    arr = np.array(lst)
    # obtain tensor object
    y_true = tf.convert_to_tensor(arr)

    visualize_confusion_matrix(y_pred_argmax, y_true)



def visualize_confusion_matrix(y_pred_argmax, y_true):
    """
    :param y_pred_arg: This is an array with values that are 0 or 1
    :param y_true: This is an array with values that are 0 or 1
    :return:
    """

    cm = tf.math.confusion_matrix(y_true, y_pred_argmax).numpy()
    con_mat_df = pd.DataFrame(cm)
    
    print(classification_report(y_pred_argmax, y_true))

    sns.heatmap(con_mat_df, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

