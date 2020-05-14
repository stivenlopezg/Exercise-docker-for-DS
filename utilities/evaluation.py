import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, \
    precision_score, recall_score, f1_score, roc_auc_score


def generate_report(y_true: list, y_pred: list):
    """
    Generate a DataFrame with the main classification metrics
    :param y_true: Observed value
    :param y_pred: Predicted value
    :return: pd.DataFrame
    """
    report = np.round(pd.DataFrame(classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)), 2).T
    return report


def confusion_matrix(y_true: list, y_pred: list, normalize: bool = True):
    """
    Generates a confusion matrix
    :param y_true: Observed value
    :param y_pred: Predicted value
    :param normalize: Boolean on normalize for each row
    :return: pd.DataFrame
    """
    if normalize:
        table = np.round(pd.crosstab(index=y_true, columns=y_pred,
                                     rownames=['Observed'], colnames=['Predicted'], normalize='index'), 2)
    else:
        table = np.round(pd.crosstab(index=y_true, columns=y_pred, rownames=['Observed'], colnames=['Predicted']), 2)
    return table


def calculate_metrics(y_true: list, y_pred: list):
    """
    Calculate the main classification metrics
    :param y_true: Observed value
    :param y_pred: Predicted value
    :return: dict
    """
    metrics = {
        'roc': np.round(roc_auc_score(y_true=y_true, y_score=y_pred), 2),
        'accuracy': np.round(accuracy_score(y_true=y_true, y_pred=y_pred), 2),
        'precision': np.round(precision_score(y_true=y_true, y_pred=y_pred), 2),
        'recall': np.round(recall_score(y_true=y_true, y_pred=y_pred), 2),
        'f1': np.round(f1_score(y_true=y_true, y_pred=y_pred), 2),
    }
    return metrics


def metrics_summary(metrics: dict):
    """
    Print a text summary of the main classification metrics
    :param metrics: dictionary with main classification metrics
    :return: string
    """
    print(f'El AUC es: {metrics["roc"]}')
    print(f'La exactitud es: {metrics["accuracy"]}')
    print(f'La precision es: {metrics["precision"]}')
    print(f'El recall es: {metrics["recall"]}')
    print(f'El puntaje F1 es: {metrics["f1"]}')
    return None
