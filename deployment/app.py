from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from preprocess_fn import noise_entity_removal, mylemmatize, text_normalization, label_to_integer, preprocess
from evaluate import evaluate, evaluate_one
from comparison import get_best_model
import os
import pickle

# Initialisation
app = Flask(__name__)
UPLOAD_FILE_PATH = './data/'
file_list = [i for i in os.listdir("./data")]

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
models_meta["topic"] = {
    "saved_tfidf": None,
    "saved_model": None
}

# prepare vectorizer and saved_model for predictions
vectorizer = pickle.load(open(models_meta["xgboost"]["saved_tfidf"], "rb"))
saved_model = pickle.load(open(models_meta["xgboost"]["saved_model"], "rb"))

topic_vectorizer = pickle.load(open(models_meta["topic"]["saved_tfidf"], "rb"))
topic_saved_model = pickle.load(open(models_meta["topic"]["saved_model"], "rb"))

@app.route('/')
def hello():
    return 'Welcome from Group Pietonium!'

'''
upload_url = 'http://127.0.0.1:5000/upload'
with open('reviews.csv', 'rb') as f: # use this in case file is too big, better to stream
    files = {'file': ('uploaded_reviews.csv', f)} # filename after upload
    #params1 = {'text_col': 'Text', 'label_col': 'Sentiment'}
    upload_response = requests.post(upload_url, files=files) #, params=params1)
upload_response.text
'''
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print('Uploading your file ...')
        uploaded_file = request.files['file']
        uploaded_filename = uploaded_file.filename
        file_list.append(uploaded_filename)
        uploaded_file.save(f"{UPLOAD_FILE_PATH + secure_filename(uploaded_filename)}")

        # preprocess immediately after upload
        print('Preprocessing your file ... (This will take a while, please wait patiently)')

        # getting the relevant column names from the uploaded dataset
        text_col_name = 'Text'
        label_col_name = 'Sentiment'
        if request.args.get('text_col'):
            text_col_name = request.args.get('text_col') # name of the text column
        if request.args.get('label_col'):
            label_col_name = request.args.get('label_col') # name of the label column

        df = pd.read_csv(UPLOAD_FILE_PATH + uploaded_filename)
        if label_col_name in df.columns:
            df = preprocess(df, label_col_name = label_col_name)
        else:
            df = preprocess(df)
        df.to_csv(UPLOAD_FILE_PATH + "processed_" + uploaded_filename)
        processed_filename = "processed_" + uploaded_filename
        file_list.append(processed_filename)

        print(f'Sucessfully uploaded and preprocessed {uploaded_filename}!')

    return "Done"

'''
requests.get('http://127.0.0.1:5000/list_files')
'''
@app.route('/list_files', methods=['GET'])
def list_files():
    print("Available Files:")
    print(file_list)
    return file_list

'''
bad_text = "This product was a complete disappointment. Poor quality and unreliable performance. Not recommended."
good_text = "It so good, I will definitely buy it again!"
prediction_url = 'http://127.0.0.1:5000/prediction'
param1 = {'text': good_text}
prediction_response = requests.get(prediction_url, params=param1)
prediction_response.text
'''
@app.route('/prediction', methods=['GET']) # user put in one sentence
def make_prediction():
    text = request.args.get('text')
    processed_text = text_normalization(noise_entity_removal(text))
    # assume final model is xgboost
    test_x = vectorizer.transform([processed_text])
    predicted_y = saved_model.predict(test_x)[0]
    return f'Predicted Sentiment: {predicted_y}'

'''
predictions_url = 'http://127.0.0.1:5000/predictions'
params2 = {'filename':'uploaded_reviews.csv'}
predictions_response = requests.get(predictions_url, params=params2)
predictions_response.text
'''
@app.route('/predictions', methods=['GET']) # user put in entire data (eg. reviews.csv)
def make_predictions():
    filename = "data/processed_uploaded_reviews.csv"
    text_col_name = 'processed_text'
    
    if request.args.get('filename'):
        input_filename = request.args.get('filename')
        if input_filename in file_list:
            filename = input_filename
            filename = UPLOAD_FILE_PATH + "processed_" + filename
            print("filename is", filename)
        else:
            print("No such file. Please upload you file via /upload")
            return "No such file"
        
    if request.args.get('text_col_name'):
        text_col_name = request.args.get('text_col_name')

    data = pd.read_csv(filename)
    test_data_feature = data[text_col_name].values.tolist()
    test_x = vectorizer.transform(test_data_feature)
    predicted_y = saved_model.predict(test_x).tolist()

    return jsonify(f'{predicted_y}')

@app.route('/get_topic', methods=['GET']) # havent try yet
def get_topic():
    text = request.args.get('text')
    processed_text = text_normalization(noise_entity_removal(text))
    test_x = topic_vectorizer.transform([processed_text])
    predicted_y = topic_saved_model.predict(test_x)[0]
    return f'Predicted Topic: {predicted_y}'

@app.route('/get_topics', methods=['GET']) # havent try yet
def get_topics():
    filename = "data/processed_uploaded_reviews.csv"
    text_col_name = 'processed_text'
    
    if request.args.get('filename'):
        input_filename = request.args.get('filename')
        if input_filename in file_list:
            filename = input_filename
            filename = UPLOAD_FILE_PATH + "processed_" + filename
            print("filename is", filename)
        else:
            print("No such file. Please upload you file via /upload")
            return "No such file"
        
    if request.args.get('text_col_name'):
        text_col_name = request.args.get('text_col_name')

    data = pd.read_csv(filename)
    test_data_feature = data[text_col_name].values.tolist()
    test_x = topic_vectorizer.transform(test_data_feature)
    predicted_y = topic_saved_model.predict(test_x).tolist()

    return jsonify(f'{predicted_y}')