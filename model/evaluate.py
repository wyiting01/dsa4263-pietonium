# load in preprocessing and relevant libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
from xgboost import XGBClassifier
import numpy as np
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay

import warnings
warnings.filterwarnings("ignore")

'''
models_meta contains filepaths for:
1. TF-IDF vectorizer for machine learning models (svm and xgboost)
2. saved model weights for both maching learning models (svm and xgboost) and deep learning model (bert)
'''
models_meta = {}
models_meta["svm"] = {
    "saved_tfidf": "uy_svm1_vectorizer.pkl",
    "saved_model": "uy_svm1.pkl"
    }
models_meta["xgboost"] = {
    "saved_tfidf": "xgboost_vectorizer.pkl",
    "saved_model": "xgboost.pkl"
    }
models_meta["bert"] = {
    "saved_model": None
    }
    
test_data_path = "../data/curated/reviews/cleaned_reviews.csv"
# svm_linear_classifier = svm.LinearSVC()
# xgb_model = XGBClassifier(random_state = 1)

'''
FUNCTIONS FOR BERT MODEL
1. bert_pipeline is a function to preprocess the dataset into the format of the bert model
2. instantiate_bert_model() is a function to create the bert architecture - weights will be updated with the weights from saved model
'''

def bert_pipeline(sentence_column, labels_column, max_len = 512):
    # Extract label values to get the size
    arr = np.array(labels_column)
    print(f"Total number of rows = {arr.size}")

    # Create 2D array to indicate which row of data the label belongs to
    labels = np.zeros((arr.size, arr.max() + 1), dtype=int)

    # Indicate the label (0 or 1) of the respective row of data
    labels[np.arange(arr.size), arr] = 1
    print(f"Label Shape: {labels.shape}")

    # Load the BERT tokenizer
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Encode our concatenated data
    print("Tokenizing sentences...")
    encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentence_column]

    # Initialise two arrays for input tensors
    print("Initialising arrays for input tensors...")
    Xids = np.zeros((len(sentence_column), max_len))
    Xmask = np.zeros((len(sentence_column), max_len))
    print(f"Input Tensors Shape: {Xids.shape}")

    print("Encoding sentences...")
    # For each text in the dataframe...
    for i, sequence in enumerate(sentence_column):
        
        # Return a dictionary containing the encoded sentence
        tokens = tokenizer.encode_plus(str(sequence), max_length = max_len, 
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

    # Combine arrays into tensorflow object
    print("Creating Tensoflow Dataset...")
    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

    # Create function to restructure the dataset
    def map_func(input_ids, masks, labels):
        return {'input_ids': input_ids, 'attention_mask': masks},labels

    # Apply map method to apply our function above to the dataset
    dataset = dataset.map(map_func)

    # Shuffle the data to prevent overfitting
    print("Shuffling dataset...")
    dataset = dataset.shuffle(100000, reshuffle_each_iteration = False)

    print("Success! Data is ready modelling!")
    return dataset

def instantiate_bert_model():
    pass

def evaluate(test_data_path = "../data/curated/reviews/cleaned_reviews.csv", target_models = ["svm", "xgboost", "bert"]): # ater running the preprocess file
    '''
    target_models: "svm", "xgboost", "bert"
    '''
    test_data = pd.read_csv(test_data_path)
    test_data_feature = test_data['processed_text'].values.tolist()
    test_y = test_data['Sentiment'].values.tolist()

    # to store every model's prediction and evalutaion metrics
    result_dict = {} 

    # evaluating all the model stated in the argument
    for model in target_models: 
        print(f"Evaluating {model} ...")
        model_result_dict = {}

        if model in ["svm", "xgboost"]: # necessary feature engineering
            vectorizer = pickle.load(open(models_meta[model]["saved_tfidf"], "rb")) # load saved tfidf vectorizer
            test_x = vectorizer.transform(test_data_feature)
            saved_model = pickle.load(open(models_meta[model]["saved_model"], "rb")) # load saved model
            
        elif model in ["bert"]: # preparation for the bert's input requirement
            test_x = bert_pipeline(test_data_feature, test_y, max_len = 512)
            saved_model = instantiate_bert_model()
            saved_model = saved_model.load_weights(models_meta[model]["saved_model"])

            lr_scheduler = PolynomialDecay(initial_learning_rate = 5e-5, end_learning_rate = 2e-5, decay_steps = num_train_steps)
            opt = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)
            loss = tf.keras.losses.CategoricalCrossentropy()
            acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
            BATCH_SIZE, EPOCHS = 32, 10
            num_train_steps = len(test_data_feature)*EPOCHS
            saved_model.compile(optimizer=opt, loss=loss, metrics=[acc, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        else:
            print("Model does not exist. Available models are 'svm', 'xgboost' and 'bert'.")
            continue

        # prediction
        predicted_y = saved_model.predict(test_x)
        model_result_dict["prediction"] = predicted_y

        # evaluation
        metric_dict = {}
        cm = confusion_matrix(test_y, predicted_y).ravel() # tn, fp, fn, tp
        accuracy = accuracy_score(test_y, predicted_y) # (tp+tn)/(tp+fp+tn+fn)
        precision = cm[3] / (cm[3] + cm[1]) # tp/(tp+fp)
        recall = cm[3] / (cm[3] + cm[2]) # tp/(tp+fn)
        metric_dict["cm"] = cm
        metric_dict["accuracy"] = accuracy
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall

        # update result dict
        model_result_dict["metrics"] = metric_dict

        result_dict[model] = model_result_dict

    return result_dict
    
print(evaluate(target_models = ["svm", "xgboost"]))