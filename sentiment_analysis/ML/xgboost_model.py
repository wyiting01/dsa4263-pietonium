#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
np.random.seed(1)

# instantiate tfidf vectorizer and return a fitted vectorizer
def xgb_instantiate_tfidf_vectorizer(x_train):
    vectorizer = TfidfVectorizer(max_df=0.8, sublinear_tf=True)
    vectorizer.fit(x_train)
    return vectorizer

# instantiate xgboost model and return fitted model
def xgb_instantiate_xgb_model(x_train, y_train, set_random_state = 1):
    xgb_model = XGBClassifier(random_state = set_random_state)
    xgb_model.fit(x_train, y_train)
    return xgb_model

# obtain classification report
def xgb_get_classification_report(actual, predicted):
    report = classification_report(actual, predicted, output_dict=True)
    return report

# print classificatoin report in a nice way
def xgb_print_classifcation_rerport(report):
    print('Positive:')
    print('---------')
    for metric, score in report['1'].items():
        print(f'{metric}: {round(score, 3)}')
        
    print('\nNegative:')
    print('---------')
    for metric, score in report['0'].items():
        print(f'{metric}: {round(score, 3)}')
        
    print('\nOverall:')
    print('--------')
    print(f"accuracy: {round(report['accuracy'], 3)}")

# obtain the plot for confusion matrix
def xgb_get_confusion_matrix(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['negative', 'positive'])
    cm_display.plot()
    plt.title('Confusion Matrix')
    plt.show()
    return cm

if __name__ == '__main__':
    # read in data, and split into train and test
    df = pd.read_csv('../../data/curated/reviews/cleaned_reviews.csv', parse_dates = ['Time'])
    x_train, x_test, y_train, y_test = train_test_split(df['processed_text'], df['Sentiment'], test_size = 0.2, random_state=4211, stratify = df['Sentiment'])

    # tfidf embeddings
    vectorizer = TfidfVectorizer(max_df=0.8, sublinear_tf=True)
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)

    # xgboost model
    xgb_model = XGBClassifier(random_state = 1)
    xgb_model.fit(train_vectors, y_train)

    # evaluation
    x_test_predicted = xgb_model.predict(test_vectors)
    report = classification_report(y_test, x_test_predicted, output_dict=True)

    print('Positive:')
    print('---------')
    for metric, score in report['1'].items():
        print(f'{metric}: {round(score, 3)}')
        
    print('\nNegative:')
    print('---------')
    for metric, score in report['0'].items():
        print(f'{metric}: {round(score, 3)}')
        
    print('\nOverall:')
    print('--------')
    print(f"accuracy: {round(report['accuracy'], 3)}")

    # confusion matrix
    cm = confusion_matrix(y_test, x_test_predicted)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['negative', 'positive'])
    cm_display.plot()
    plt.title('Confusion Matrix')
    plt.show()

    # save TF-IDF vectorizer and XGBoost model
    pickle.dump(vectorizer, open('../../model/xgboost_vectorizer.pkl', 'wb'))
    pickle.dump(xgb_model, open('../../model/xgboost.pkl', 'wb'))
