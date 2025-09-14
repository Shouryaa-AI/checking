# src/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """Plots a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_correlation_matrix(correlation_df, title='Correlation Matrix'):
    """Plots a correlation matrix."""
    if not correlation_df.empty:
        correlation_matrix = correlation_df.corr()
        plt.figure(figsize=(14, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(title)
        plt.show()
    else:
        print("No data for correlation matrix plot.")

def plot_sentiment_vs_metric(df, sentiment_col, metric_col, title, hue_col=None):
    """Plots sentiment against a given metric."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=sentiment_col, y=metric_col, hue=hue_col)
    plt.title(title)
    plt.xlabel(sentiment_col)
    plt.ylabel(metric_col)
    plt.show()

def plot_boxplot_sentiment_metric(df, sentiment_label_col, metric_col, title, order=None):
    """Plots a boxplot of a metric by sentiment label."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=sentiment_label_col, y=metric_col, order=order)
    plt.title(title)
    plt.xlabel(sentiment_label_col)
    plt.ylabel(metric_col)
    plt.show()

def plot_barplot_topic_metric(df, topic_id_col, metric_col, title):
    """Plots a barplot of a metric by dominant topic."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.dropna(subset=[topic_id_col]), x=topic_id_col, y=metric_col)
    plt.title(title)
    plt.xlabel('Dominant Topic ID')
    plt.ylabel(f'Average {metric_col}')
    plt.show()
