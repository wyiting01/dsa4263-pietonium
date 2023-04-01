# load in preprocessing and relevant libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn import svm

'''
models_meta contains filepaths for:
1. TF-IDF vectorizer for machine learning models (svm and xgboost)
2. saved model weights for both maching learning models (svm and xgboost) and deep learning model (bert)
'''
models_meta = {}
models_meta["svm"] = {
    "saved_tfidf": "svm_vectorizer.pkl",
    "saved_model": "svm.pkl"
    }
models_meta["xgboost"] = {
    "saved_tfidf": "xgboost_vectorizer.pkl",
    "saved_model": "xgboost.pkl"
    }
models_meta["bert"] = {
    "saved_model": None
    }
    
test_data_path = "../data/curated/reviews/cleaned_reviews.csv"
svm_linear_classifier = svm.LinearSVC()
xgb_model = XGBClassifier(random_state = 1)

def evaluate(test_data_path, target_models = ["svm", "xgboost", "bert"]): # ater running the preprocess file
    '''
    target_models: "svm", "xgboost", "bert"
    '''
    test_data = pd.read_csv(test_data_path)
    test_data_feature = test_data['processed_text'].values.to_list()
    test_y = test_data['Sentiment'].values.to_list()

    # to store every model's prediction and evalutaion metrics
    result_dict = {} 

    # evaluating all the model stated in the argument
    for model in target_models: 
        print(f"Evaluating {model} ...")
        model_result_dict = {}

        if model in ["svm", "xgboost"]: # necessary feature engineering
            vectorizer = pickle.load(models_meta[model]["saved_tfidf"]) # load saved tfidf vectorizer
            test_x = vectorizer.transform(test_data_feature)
            
        elif model in ["bert"]: # preparation for the bert's input requirement
            test_x = None
            pass

        else:
            print("Model does not exist. Available models are 'svm', 'xgboost' and 'bert'.")
            continue

        # prediction
        saved_model = pickle.load(models_meta[model]["saved_model"]) # load saved model
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
    

