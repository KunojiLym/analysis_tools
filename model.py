from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, log_loss

import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(model, X, y, ticklabels=None, threshold=0.5, show_metrics=False, normalize=None):
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)
    
    y_predict = (model.predict_proba(X)[:, 1] >= threshold)
    confusion = confusion_matrix(y, y_predict, normalize=normalize)
    plt.figure(dpi=80)

    if ticklabels == None:
        sns.heatmap(confusion, cmap=plt.cm.get_cmap('Blues'), annot=True, square=True, fmt='g')
    else:
        sns.heatmap(confusion, cmap=plt.cm.get_cmap('Blues'), annot=True, square=True, fmt='g',
            xticklabels=ticklabels,
            yticklabels=ticklabels)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if show_metrics:
        plt.text(3, 0,'Precision:')
        plt.text(3.5, 0, '{:.2}'.format(precision_score(y, y_predict)))
        plt.text(3, 0.2,'Recall:')
        plt.text(3.5, 0.2, '{:.2}'.format(recall_score(y, y_predict)))
        plt.text(3, 0.4,'Log Loss:')
        plt.text(3.5, 0.4, '{:.2}'.format(log_loss(y, y_predict)))