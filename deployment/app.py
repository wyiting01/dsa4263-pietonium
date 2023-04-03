from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from preprocess_fn import noise_entity_removal, mylemmatize, text_normalization, label_to_integer
from evaluate import evaluate, evaluate_one
from comparison import get_best_model

app = Flask(__name__)
UPLOAD_FILE_PATH = './data/'

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
    "saved_model": None
}

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
        df['processed_text'] = df[text_col_name].apply(lambda x:noise_entity_removal(x))
        df['processed_text'] = df['processed_text'].apply(lambda x:text_normalization(x))
        df[label_col_name] = df[label_col_name].apply(lambda x:label_to_integer(x))
        df.to_csv(UPLOAD_FILE_PATH + "processed_" + uploaded_filename)

        print(f'Sucessfully uploaded and preprocessed {uploaded_filename}!')

    return "Done"

'''
prediction_url = 'http://127.0.0.1:5000/prediction'
param1 = {'text': 'this is just some random text for testing', 'actual_label':'positive', 'preferred_model': "xgboost svm"}
prediction_response = requests.get(prediction_url, params=param1)
prediction_response.text
'''
@app.route('/prediction', methods=['GET']) # user put in one sentence
def make_prediction():
    preferred_models = "xgboost svm bert"
    text, actual_label = request.args.get('text'), request.args.get('actual_label')

    if request.args.get('preferred_models'):
        preferred_models = request.args.get('preferred_models')

    if " " in preferred_models:
        preferred_models_list = preferred_models.split()
    else:
        preferred_models_list = [preferred_models]

    processed_text = text_normalization(noise_entity_removal(text))
    processed_actual_label = label_to_integer(actual_label)

    evaluation_output = evaluate_one(processed_text, processed_actual_label, models_meta, target_models = preferred_models_list)
    print(evaluation_output)
    return jsonify(f'{evaluation_output}')

'''
predictions_url = 'http://127.0.0.1:5000/predictions'
predictions_response = requests.get(predictions_url)
predictions_response.text
'''
@app.route('/predictions', methods=['GET']) # user put in entire data (eg. reviews.csv)
def make_predictions():
    preferred_models = "xgboost svm bert"
    filename = "data/processed_uploaded_reviews.csv"
    
    if request.args.get('filename'):
        filename = request.args.get('filename')
    if request.args.get('preferred_models'):
        preferred_models = request.args.get('preferred_models')

    if " " in preferred_models:
        preferred_models_list = preferred_models.split()
    else:
        preferred_models_list = [preferred_models]
        
    evaluation_output = evaluate(models_meta, filename, preferred_models_list)
    print(evaluation_output)

    print(get_best_model(evaluation_output))
    return "done"

@app.route('/get_topic', methods=['GET'])
def get_topic():
    pass

@app.route('/get_topics', methods=['GET'])
def get_topics():
    pass