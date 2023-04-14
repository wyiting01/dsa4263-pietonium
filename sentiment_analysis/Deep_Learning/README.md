<h1>Deep Learning repository</h1>
<b></b>
This directory is used for storing training scripts of deep learning models being experimented and their associated requirements. 

We fine-tuned two versions of BERT:
  1. **bert-based-cased** from Hugging Face using **Tensorflow**
  2. **distilbert-base-uncased** from Hugging Face using **Pytorch**

The second one is used for presentation and behind scoring API

<h3>Fine-Tune DistilBERT with Pytorch</h3>
  1. **distilbert.py**: Contains functions that abstract away the low-level code during evaluation and scoring, which are used both for
presentation and scoring api behind the web app. It also contains configurations of training arguments
  2. **distilbert_train_modules.py**: Contains modularized components (functions) of the training pipeline, making use of some helper functions in *distilbert.py*
  3. **distilbert_train.ipynb**: The jupyter notebook that import modules from *distilbert.py* and *distilbert_train_modules.py* to perform model fine-tuning based on new data. This script was run with GPU backend (Google Colab or AWS instance)
  4. **DistilBERT1_requirements.txt**: Contains Python packages for virtual environment that runs the model training.


