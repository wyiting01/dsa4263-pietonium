<h1>Deep Learning repository</h1>
<b></b>
This directory is used for storing training scripts of deep learning models being experimented and their associated requirements. 

We fine-tuned two versions of BERT:
  1. **bert-based-cased** from Hugging Face using **Tensorflow**
  2. **distilbert-base-uncased** from Hugging Face using **Pytorch**

The second one is used for presentation and behind scoring API

<h2>Fine-Tune DistilBERT with Pytorch</h2>

  1. **distilbert.py**: Contains functions that abstract away the low-level code during evaluation and scoring, which are used both for
presentation and scoring api behind the web app. It also contains configurations of training arguments

  2. **distilbert_train_modules.py**: Contains modularized components (functions) of the training pipeline, making use of some helper functions in *distilbert.py*
  
  4. **distilbert_train.ipynb**: The jupyter notebook importing modules from *distilbert.py* and *distilbert_train_modules.py* to perform model fine-tuning based on reviews data. This script was run with GPU backend (Google Colab or AWS instance)
  
  5. **DistilBERT1_requirements.txt**: Contains Python packages for virtual environment that runs the model training.

<h3>distilbert.py</h3>

***General Helper Functions***

**read_data(path)**
	
  Description: read cleaned text data with columns 'Sentiment' and 'processed text' and return a Pandas Dataframe
	
  Input: path to the csv data, type <str>
	
  Output: pd.DataFrame

**make_dataset(df)**
	
  Description: Read the cleaned dataframe and return the Dataset object that is fed to model during training
	
  Input: pd.DataFrame with columns ['processed_text', 'Sentiment']
	
  Output: a datasets.Dataset object

**load_distilbert_model()**
	
  Description: load pretrained model from Hugging Face and return a tuple of (model, tokenizer)
	
  Output: tuple of (model, tokenizer)

**noise_remove_helper(raw_text)**
	
  Description: Read in the raw text and perform noise removal (remove digits, html tags, non-word characters, extra spaces, words less than 3 characters)
	
  Input: str (raw text)
	
  Output: str (cleaned text)

**label_to_integer_helper(label)**
	
  Description: Encode 'positive' as 1 and 'negative' as 0
	
  Input: str
	
  Output: int

**id_2_label(model, id)**
	
  Description: Encode predicted 1 and 0 as 'Positive' and 'Negative' respectively
	
  Input: int
	
  Output: str

For App and Evaluation

**scoring_single_review(raw_review, model, tokenizer):**
	
  Description: Read in a raw review and give back predicted sentiment (1/0) and its associated probability
	
  Output: (predicted_sentiment, predicted_sentiment_probability) -> (int,float)

**prepare_inference_data(filename):**
	
  Description: read a csv file containing data of columns `Text` and `Time` and add a pre-processed text column `processed_text`
	
  Input: str
	
  Output: pd.DataFrame of columns: `Text`, `Time`, `processed_text`

**scoring_file_thread(filename, model, tokenizer):**
	
  Description: Read the filename and return the pd.DataFrame of columns ['Time', 'Text', 'predicted_sentiment', 'predicted_sentiment_probability'],
	using multi-threading to handle reviews

**scoring_file_thread_df(df, model, tokenizer):**
	
  Description: Do the same as *scoring_file_thread* but take in a pd.DataFrame as input instead

**scoring_file_dummy(filename, model, tokenizer):**
	
  Description: Read the filename and return the pd.DataFrame of columns ['Time', 'Text', 'predicted_sentiment', 'predicted_sentiment_probability'],
	using traditional for-loop to handle reviews.

**scoring_file_dummy_df(df, model, tokenizer):**
	
  Description: Do the same as *scoring_file_dummy* but take in a pd.DataFrame as input instead

**visualize_confusion_matrix(predicted_labels, real_labels):**
	
  Description: Take in an array-like objects shape (n,) of predicted labels and real labels, and plot a confusion matrix

**confusion_matrix(predicted_labels, real_labels):**
	
  Description: Take in an array-like objects shape (n,) of predicted labels and real labels, and output a sklearn classification_report

**predict(input, model, tokenizer):**
	
  Description: This function is for scoring multiple reviews or single reviews, return the predicted sentiment and its associated probability
	
  Input - Output
		
    If single review (str) -> return (predicted_class_id, predicted_class_prob) by calling *scoring_single_review*
        
    If array-like -> return dict of {'preds': predicted_labels, 'preds_prob': predicted_label_probs}
