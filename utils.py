"""Utility functions for Decision Tree model and data preprocessing.

This module provides helper functions for:
- Data visualization and analysis
- Model evaluation and comparison
- Hyperparameter optimization
- Results visualization and reporting

Author: Mihir Brahmaniya
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', figsize=(8, 6)):
    """Plot confusion matrix heatmap.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        title (str): Title for the plot.
        figsize (tuple): Figure size.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10, figsize=(10, 6)):
    """Plot feature importance from tree-based models.
    
    Args:
        model: Trained sklearn model with feature_importances_ attribute.
        feature_names (list): List of feature names.
        top_n (int): Number of top features to display. Default is 10.
        figsize (tuple): Figure size.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, model_name='Model', figsize=(8, 6)):
    """Plot ROC curve.
    
    Args:
        y_true (array-like): True binary labels.
        y_pred_proba (array-like): Predicted probabilities for positive class.
        model_name (str): Name of the model for legend. Default is 'Model'.
        figsize (tuple): Figure size.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_class_distribution(y, title='Class Distribution', figsize=(8, 5)):
    """Plot class distribution as bar chart.
    
    Args:
        y (array-like): Target variable.
        title (str): Title for the plot.
        figsize (tuple): Figure size.
    """
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=figsize)
    plt.bar(unique, counts, color=['steelblue', 'coral'])
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(title)
    for i, v in enumerate(counts):
        plt.text(unique[i], v + 0.1, str(v), ha='center', va='bottom')
    plt.show()


def print_model_metrics(y_true, y_pred, y_pred_proba=None, model_name='Model'):
    """Print comprehensive model evaluation metrics.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_pred_proba (array-like, optional): Predicted probabilities.
        model_name (str): Name of the model.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    print(f"\n{'='*50}")
    print(f"Model Evaluation: {model_name}")
    print(f"{'='*50}")
    
    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            print(f"ROC-AUC:   {roc_auc:.4f}")
        except:
            pass
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print(f"{'='*50}\n")


def compare_models(models_dict, X_test, y_test):
    """Compare multiple models on test set.
    
    Args:
        models_dict (dict): Dictionary with model names as keys and models as values.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
    """
    from sklearn.metrics import accuracy_score
    
    results = {}
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name:.<30} {accuracy:.4f}")
    
    best_model = max(results, key=results.get)
    print("="*50)
    print(f"Best Model: {best_model} ({results[best_model]:.4f})")
    print("="*50 + "\n")
    
    return results


def plot_learning_curve(train_sizes, train_scores, val_scores, title='Learning Curve', figsize=(10, 6)):
    """Plot learning curve showing training vs validation scores.
    
    Args:
        train_sizes (array-like): Training set sizes.
        train_scores (array-like): Training scores.
        val_scores (array-like): Validation scores.
        title (str): Title for the plot.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_scores, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, val_scores, 'o-', color='g', label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    print("Utility functions module loaded successfully!")
    print("Available functions:")
    print("- plot_confusion_matrix()")
    print("- plot_feature_importance()")
    print("- plot_roc_curve()")
    print("- plot_class_distribution()")
    print("- print_model_metrics()")
    print("- compare_models()")
    print("- plot_learning_curve()")
