from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from preprocess_fn import noise_entity_removal, mylemmatize, text_normalization, label_to_integer
from evaluate import evaluate, evaluate_one

app = Flask(__name__)
UPLOAD_FILE_PATH = './data/'

@app.route('/')
def hello():
    return 'Welcome from Group Pietonium!'

'''
upload_url = 'http://127.0.0.1:5000/upload'
with open('xgboost.pkl', 'rb') as f:
    files = {'file': ('uploaded_xgboost.pkl', f)}
    upload_response = requests.post(upload_url, files=files)
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
            text_col_name = request.args.get('text_col') # Request's argument
        if request.args.get('label_col'):
            label_col_name = request.args.get('label_col') # Request's argument

        df = pd.read_csv(UPLOAD_FILE_PATH + uploaded_filename)
        df['processed_text'] = df[text_col_name].apply(lambda x:noise_entity_removal(x))
        df['processed_text'] = df['processed_text'].apply(lambda x:text_normalization(x))
        df[label_col_name] = df[label_col_name].apply(lambda x:label_to_integer(x))
        df.to_csv(UPLOAD_FILE_PATH + "processed_" + uploaded_filename)

        print(f'Sucessfully uploaded and preprocessed {uploaded_filename}!')

    return "Done"

@app.route('/prediction', methods=['GET']) # user put in one sentence
def make_prediction():
    text, actual_label, preferred_model = request.args.get('text'), request.args.get('actual_label'), request.args.get('preferred_model')

    if " " in preferred_model:
        preferred_model_list = preferred_model.split()
    else:
        preferred_model_list = [preferred_model]

    processed_text = text_normalization(noise_entity_removal(text))
    processed_actual_label = label_to_integer(actual_label)

    evaluation = evaluate_one(processed_text, processed_actual_label, target_models = preferred_model_list)
    return jsonify(f'{evaluation}')

@app.route('/predictions', methods=['GET']) # user put in entire data (eg. reviews.csv)
def make_predictions():
    pass


