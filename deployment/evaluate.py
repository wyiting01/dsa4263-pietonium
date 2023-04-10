# load in preprocessing and relevant libraries
import pathlib
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

THIS_DIR = pathlib.Path(__file__).resolve()
PROJ_DIR = THIS_DIR.parents[1]
sys.path.append(PROJ_DIR.as_posix())

from sentiment_analysis.Deep_Learning.bert import (build_layers, read_data, tokenization)
    
test_data_path = "../data/curated/reviews/cleaned_reviews.csv"
# svm_linear_classifier = svm.LinearSVC()
# xgb_model = XGBClassifier(random_state = 1)

'''
FUNCTIONS FOR BERT MODEL
1. bert_pipeline is a function to preprocess the dataset into the format of the bert model
2. instantiate_bert_model() is a function to create the bert architecture - weights will be updated with the weights from saved model
'''

def bert_pipeline(test_data):
    cleaned_df = test_data.rename(columns = {'Sentiment' : 'labels', 'processed_text' : 'text'})
    cleaned_df = cleaned_df[['text','labels']]
    X_testids, X_testmask = tokenization(cleaned_df)
    X_test = [X_testids, X_testmask]
    return X_test

def instantiate_bert_model():
    model = build_layers()
    return model

def evaluate(models_meta, test_data, target_models = ["svm", "xgboost", "bert"]): # ater running the preprocess file
    '''
    target_models: "svm", "xgboost", "bert"
    '''
    # test_data = pd.read_csv(test_data_path)
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
            test_x = bert_pipeline(test_data)
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

'''
This function is to make predictions on the test data with the models specified in the parameter 'target_models'. It will the output
a dictionary which contains every model and its respective results and metrics. The output can then be used to compare the performance
of these models.
'''
def evaluate_with_models(x_test, y_test, target_models):
    result_dict = {}
    for model_name, model in target_models:
        model_result_dict = {}
        if model_name != "bert":
            predicted_y = model.predict(x_test)
            predicted_prob_y = model.predict_proba(x_test)
        else:
            predicted_prob_y = tf.nn.softmax(model.predict(x_test))
            predicted_y = tf.math.argmax(predicted_prob_y, axis=1)

        model_result_dict["prediction"] = predicted_y
        model_result_dict["prediction_prob"] = predicted_prob_y

        metric_dict = {}
        cm_2d = confusion_matrix(y_test, predicted_y)
        cm = cm_2d.ravel() # tn, fp, fn, tp
        accuracy = accuracy_score(y_test, predicted_y) # (tp+tn)/(tp+fp+tn+fn)
        precision = cm[3] / (cm[3] + cm[1]) # tp/(tp+fp)
        recall = cm[3] / (cm[3] + cm[2]) # tp/(tp+fn)
        metric_dict["cm"] = cm_2d
        metric_dict["accuracy"] = accuracy
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall

        # update result dict
        model_result_dict["metrics"] = metric_dict

        # update to result dict
        result_dict[model_name] = model_result_dict
    return result_dict