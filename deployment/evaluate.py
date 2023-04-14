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

sys.path.append("../")
from sentiment_analysis.Deep_Learning.distilbert import predict as db_predict

warnings.filterwarnings("ignore")

'''
This function is to make predictions on the test data with the models specified in the parameter 'target_models'. It will the output
a dictionary which contains every model and its respective results and metrics. The output can then be used to compare the performance
of these models.
'''
def evaluate_with_models(x_test, bert_x_test, y_test, target_models):
    result_dict = {}
    for model_name, model in target_models:
        print(f'Evaluating {model_name} ...')
        model_result_dict = {}
        if model_name != "bert":
            predicted_y = model.predict(x_test)
            predicted_prob_y = model.predict_proba(x_test)
        else:
            model_hf, tokenizer_hf = model
            predictions = db_predict(bert_x_test, model=model_hf, tokenizer=tokenizer_hf)
            predicted_y = np.array(predictions['preds'])
            predicted_prob_y = np.array(predictions['preds_prob'])

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