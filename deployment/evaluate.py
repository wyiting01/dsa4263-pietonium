# load in preprocessing and relevant libraries
import pathlib
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# THIS_DIR = pathlib.Path(__file__).resolve()
# PROJ_DIR = THIS_DIR.parents[1]
# sys.path.append(PROJ_DIR.as_posix())

# from sentiment_analysis.Deep_Learning.bert import (build_layers, read_data,
#                                                    tokenization)


'''
models_meta contains filepaths for:
1. TF-IDF vectorizer for machine learning models (svm and xgboost)
2. saved model weights for both maching learning models (svm and xgboost) and deep learning model (bert)
'''
models_meta = {}
models_meta["svm"] = {
    "saved_tfidf": "saved_models/uy_svm1_vectorizer.pkl",
    "saved_model": "saved_models/uy_svm1.pkl"
    }
models_meta["xgboost"] = {
    "saved_tfidf": "saved_models/xgboost_vectorizer.pkl",
    "saved_model": "saved_models/xgboost.pkl"
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

# def bert_pipeline(test_data_path):
#     cleaned_df = read_data(test_data_path)
#     X_testids, X_testmask = tokenization(cleaned_df)
#     X_test = [X_testids, X_testmask]
#     return X_test

# def instantiate_bert_model():
#     model = build_layers()
#     return model

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
        print(f"Evaluating {model.upper()} ...")
        model_result_dict = {}

        if model in ["svm", "xgboost"]: # necessary feature engineering
            vectorizer = pickle.load(open(models_meta[model]["saved_tfidf"], "rb")) # load saved tfidf vectorizer
            test_x = vectorizer.transform(test_data_feature)
            saved_model = pickle.load(open(models_meta[model]["saved_model"], "rb")) # load saved model
            
        elif model in ["bert"]: # preparation for the bert's input requirement
            # test_x = bert_pipeline(test_data_path)
            # saved_model = instantiate_bert_model()
            # saved_model = saved_model.load_weights(models_meta[model]["saved_model"])

            # lr_scheduler = PolynomialDecay(initial_learning_rate = 5e-5, end_learning_rate = 2e-5, decay_steps = num_train_steps)
            # opt = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)
            # loss = tf.keras.losses.CategoricalCrossentropy()
            # acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
            # BATCH_SIZE, EPOCHS = 32, 10
            # num_train_steps = len(test_data_feature)*EPOCHS
            # saved_model.compile(optimizer=opt, loss=loss, metrics=[acc, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            pass

        else:
            print("Model does not exist. Available models are 'svm', 'xgboost' and 'bert'.")
            continue

        # prediction
        predicted_y = saved_model.predict(test_x)
        model_result_dict["prediction"] = predicted_y

        # evaluation
        metric_dict = {}
        cm_2d = confusion_matrix(test_y, predicted_y)
        cm = cm_2d.ravel() # tn, fp, fn, tp
        accuracy = accuracy_score(test_y, predicted_y) # (tp+tn)/(tp+fp+tn+fn)
        precision = cm[3] / (cm[3] + cm[1]) # tp/(tp+fp)
        recall = cm[3] / (cm[3] + cm[2]) # tp/(tp+fn)
        metric_dict["cm"] = cm_2d
        metric_dict["accuracy"] = accuracy
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall

        # update result dict
        model_result_dict["metrics"] = metric_dict

        result_dict[model] = model_result_dict

    return result_dict

def evaluate_one(text, actual_label, target_models = ["svm", "xgboost", "bert"]): # ater running the preprocess file
    '''
    target_models: "svm", "xgboost", "bert"
    '''

    # to store every model's prediction and evalutaion metrics
    result_dict = {} 

    # evaluating all the model stated in the argument
    for model in target_models: 
        print(f"Evaluating {model.upper()} ...")

        if model in ["svm", "xgboost"]: # necessary feature engineering
            vectorizer = pickle.load(open(models_meta[model]["saved_tfidf"], "rb")) # load saved tfidf vectorizer
            test_x = vectorizer.transform([text])
            saved_model = pickle.load(open(models_meta[model]["saved_model"], "rb")) # load saved model
            
        elif model in ["bert"]: # preparation for the bert's input requirement
            # test_x = bert_pipeline(test_data_path)
            # saved_model = instantiate_bert_model()
            # saved_model = saved_model.load_weights(models_meta[model]["saved_model"])

            # lr_scheduler = PolynomialDecay(initial_learning_rate = 5e-5, end_learning_rate = 2e-5, decay_steps = num_train_steps)
            # opt = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)
            # loss = tf.keras.losses.CategoricalCrossentropy()
            # acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
            # BATCH_SIZE, EPOCHS = 32, 10
            # num_train_steps = len(test_data_feature)*EPOCHS
            # saved_model.compile(optimizer=opt, loss=loss, metrics=[acc, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            pass

        else:
            print("Model does not exist. Available models are 'svm', 'xgboost' and 'bert'.")
            continue

        # prediction
        predicted_y = saved_model.predict(test_x)[0]
        result_dict[model] = predicted_y

    return result_dict