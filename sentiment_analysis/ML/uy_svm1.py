import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, cross_val_score

def svm_instantiate_tfidf_vectorizer(x_train):
    vectorizer = TfidfVectorizer(max_df=0.8, sublinear_tf=True)
    vectorizer.fit(x_train)
    return vectorizer

def svm_instantiate_svm_model(x_train, y_train, set_random_state = 1):
    svm_model = svm.LinearSVC(random_state = set_random_state)
    svm_model.fit(x_train, y_train)
    return svm_model

def svm_get_classification_report(actual, predicted):
    report = classification_report(actual, predicted, output_dict=True)
    return report

def svm_print_classifcation_rerport(report):
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

def svm_get_confusion_matrix(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['negative', 'positive'])
    cm_display.plot()
    plt.title('Confusion Matrix')
    plt.show()
    return cm

if __name__ == '__main__':
    # Import cleaned data 1 for SVM from curated
    full_df = pd.read_csv('../../data/curated/reviews/svm/uy_cleaned_reviews1.csv')
    full_df.head()

    # Train-Test Split -> leave out 20% data for testing
    from sklearn.model_selection import train_test_split
    full_label = full_df['Sentiment']
    full_text = full_df['Text']
    # full_text = np.array(full_df['Text']).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(full_text, full_label, test_size = 0.2, random_state=4211)

    # Feature Engineering
    from sklearn.feature_extraction.text import TfidfVectorizer

    tf_idf_vectorizer = TfidfVectorizer(max_df=0.5, sublinear_tf=True)
    train_vectors = tf_idf_vectorizer.fit_transform(X_train)

    # Fit SVM & CV
    import time
    from sklearn import svm
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import KFold, cross_val_score


    ## CV estimation
    svm_linear_classifier = svm.LinearSVC()
    k_folds = KFold(n_splits=5)
    val_scores = cross_val_score(svm_linear_classifier, train_vectors, y_train, cv=k_folds)

    print(f"Cross validation scores: {val_scores}") # [0.90355913 0.88289323 0.86796785 0.88518944 0.88978186]
    print(f"Average CV scores: {val_scores.mean()}") # 0.88587


    starttrain = time.time()
    svm_linear_classifier.fit(train_vectors, y_train)
    endtrain = time.time()


    # Evaluation + Prediction
    test_vectors = tf_idf_vectorizer.transform(X_test)
    startpred = time.time()
    svm_preds_linear = svm_linear_classifier.predict(test_vectors)
    endpred = time.time()

    print(f"Training time: {endtrain - starttrain}")
    print(f"Prediction time: {endpred - startpred}")

    # results

    report = classification_report(y_test, svm_preds_linear, output_dict=True)
    print('positive: ', report['1'])
    print('negative: ', report['0'])
    print('accuracy: ', report['accuracy']) # 0.895


    cm = confusion_matrix(y_test, svm_preds_linear)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['negative', 'positive'])
    cm_display.plot()
    plt.show()


    ## SAVE MODEL ##

    import pickle
    import os
    os.makedirs('../../model/', exist_ok=True)

    # pickling the TF-IDF vectorizer
    pickle.dump(tf_idf_vectorizer, open('../../model/uy_svm1_vectorizer.pkl', 'wb'))

    # pickling the SVM Linear Classifier
    pickle.dump(svm_linear_classifier, open('../../model/uy_svm1.pkl', 'wb'))

