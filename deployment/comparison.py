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
    best_metric_model_results = evaluation_output[best_metric_model_name]

    print(f"Best Model: {best_metric_model_name.upper()} with {metric} = {model_metric_list[best_metric_model_index][1]}")
    
    if output_cm:
        cm = evaluation_output[best_metric_model_name]["metrics"]["cm"]
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['negative', 'positive'])
        cm_display.plot()
        plt.title(f'Confusion Matrix for {best_metric_model_name.upper()}')
        txt = "Please close this window to finish executing the code"
        plt.xlabel("Predicted label\n(Close this window to proceed)")
        plt.show()
        return best_metric_model_name, best_metric_model_results
    
    return best_metric_model_name, best_metric_model_results