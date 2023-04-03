import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_best_model(evaluation_output, metric = "accuracy", output_cm = True):
    '''
    evalutation: output from evaluate.py
    metrics: accuracy, precision, recall

    returns the evaluation for the best model
    '''
    best_model_index = None
    model_metric_list = []
    
    for i, (model, model_evaluation) in enumerate(evaluation_output.items()):
        curr_model_metric = evaluation_output[model]["metrics"][metric]
        print(f'{model}: {curr_model_metric}')
        model_metric_list.append((model, curr_model_metric))
        
    model_metric_list = np.array(model_metric_list)
    best_metric_model_index = np.argmax(model_metric_list[:, 1])
    best_metric_model_name = model_metric_list[best_metric_model_index][0]
    
    if output_cm:
        predicted_labels = evaluation_output[best_metric_model_name]["prediction"]
        cm = confusion_matrix(actual_labels, predicted_labels) # actual_labels from global environment
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['negative', 'positive'])
        cm_display.plot()
        plt.title(f'Confusion Matrix for {best_metric_model_name.upper()}')
        plt.show()
    
    return best_metric_model_name, model_metric_list