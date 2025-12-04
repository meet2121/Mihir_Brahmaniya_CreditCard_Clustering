"""Decision Tree Classification Model

This module implements a Decision Tree classifier for the Play prediction task.
It includes data preprocessing, model training, hyperparameter tuning, and evaluation.

Author: Mihir Brahmaniya
Date: 2024
Version: 1.0
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class DecisionTreeModel:
    """Decision Tree classifier for Play prediction.
    
    This class handles the entire workflow of building a decision tree model,
    including data preprocessing, training, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self, random_state=42):
        """Initialize the DecisionTreeModel.
        
        Args:
            random_state (int): Random seed for reproducibility. Default is 42.
        """
        self.random_state = random_state
        self.model = None
        self.pipeline = None
        self.categorical_features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath):
        """Load dataset from CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        data = {
            'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
            'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
            'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
            'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
            'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
        }
        self.df = pd.DataFrame(data)
        self.df.index = range(1, len(self.df) + 1)
        print(f"Dataset loaded. Shape: {self.df.shape}")
        return self.df
        
    def display_data(self):
        """Display first rows and class distribution."""
        print("\n=== Dataset Overview ===")
        print(self.df.head())
        print("\n=== Class Distribution ===")
        print(self.df['Play'].value_counts())
        
    def preprocess_data(self, test_size=0.25, stratify=True):
        """Preprocess and split the dataset.
        
        Args:
            test_size (float): Proportion of dataset for testing. Default is 0.25.
            stratify (bool): Whether to stratify split by target variable. Default is True.
        """
        X = self.df.drop(columns=['Play'])
        y = self.df['Play']
        
        stratify_param = y if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_param
        )
        
        print(f"\nTrain size: {self.X_train.shape}")
        print(f"Test size: {self.X_test.shape}")
        
    def build_pipeline(self):
        """Build preprocessing and model pipeline."""
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(sparse_output=False, drop='first'), self.categorical_features)
            ],
            remainder='drop'
        )
        
        dt = DecisionTreeClassifier(random_state=self.random_state)
        self.pipeline = make_pipeline(self.preprocessor, dt)
        print("\nPipeline built successfully.")
        
    def hyperparameter_tuning(self, cv=3):
        """Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            cv (int): Number of cross-validation folds. Default is 3.
        """
        param_grid = {
            'decisiontreeclassifier__max_depth': [None, 1, 2, 3, 4],
            'decisiontreeclassifier__min_samples_leaf': [1, 2]
        }
        
        grid = GridSearchCV(
            self.pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        grid.fit(self.X_train, self.y_train)
        print(f"\n=== Best Hyperparameters ===")
        print(f"Best params: {grid.best_params_}")
        print(f"Best CV score: {grid.best_score_}")
        
        self.model = grid.best_estimator_
        return grid
        
    def evaluate(self):
        """Evaluate model on test set."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n=== Model Evaluation ===")
        print(f"Test Accuracy: {accuracy:.2f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return accuracy
        
    def visualize_tree(self, figsize=(14, 8), fontsize=10):
        """Visualize the trained decision tree.
        
        Args:
            figsize (tuple): Figure size for the plot.
            fontsize (int): Font size for tree labels.
        """
        trained_tree = self.model.named_steps['decisiontreeclassifier']
        ohe = self.model.named_steps['columntransformer'].named_transformers_['ohe']
        feature_names = list(ohe.get_feature_names_out(self.categorical_features))
        
        plt.figure(figsize=figsize)
        plot_tree(
            trained_tree,
            feature_names=feature_names,
            class_names=trained_tree.classes_,
            filled=True,
            rounded=True,
            fontsize=fontsize
        )
        plt.title('Decision Tree: Play vs Not Play')
        plt.show()
        
    def predict_sample(self, sample_data):
        """Make predictions on sample data.
        
        Args:
            sample_data (dict or pd.DataFrame): Input sample(s) for prediction.
            
        Returns:
            tuple: Predictions and prediction probabilities.
        """
        if isinstance(sample_data, dict):
            sample_data = pd.DataFrame([sample_data])
            
        prediction = self.model.predict(sample_data)
        probabilities = self.model.predict_proba(sample_data)
        
        print(f"\nSample: {sample_data.iloc[0].to_dict()}")
        print(f"Prediction: {prediction[0]}")
        print(f"Prediction probabilities: {probabilities[0]}")
        
        return prediction, probabilities
        
    def save_model(self, filepath='play_decision_tree_pipeline.pkl'):
        """Save trained model to file.
        
        Args:
            filepath (str): Path to save the model.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nSaved model to {filepath}")
        
    def load_model(self, filepath='play_decision_tree_pipeline.pkl'):
        """Load model from file.
        
        Args:
            filepath (str): Path to load the model from.
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"\nLoaded model from {filepath}")
        
    def run_full_pipeline(self, save_model=True):
        """Run the complete training pipeline.
        
        Args:
            save_model (bool): Whether to save the trained model. Default is True.
        """
        print("\n" + "="*50)
        print("DECISION TREE MODEL TRAINING PIPELINE")
        print("="*50)
        
        self.load_data(None)
        self.display_data()
        self.preprocess_data()
        self.build_pipeline()
        self.hyperparameter_tuning()
        self.evaluate()
        self.visualize_tree()
        
        if save_model:
            self.save_model()
            
        print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    # Initialize and run the model
    model = DecisionTreeModel(random_state=42)
    model.run_full_pipeline(save_model=True)
    
    # Make prediction on a sample
    sample = {
        'Outlook': 'Sunny',
        'Temperature': 'Cool',
        'Humidity': 'High',
        'Windy': False
    }
    model.predict_sample(sample)
