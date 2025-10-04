'''
Simple ML training script with MLOps best practices (amishjash22)
'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import json
from datetime import datetime

def load_data():
    '''Load the iris dataset'''
    print('Loading data...')
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target,name='target')
    return X,y

def prepare_data(X, y, test_size=0.2, random_state=42):
    '''Split data into train and test sets'''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=100, max_depth=3):
    '''Train Random Forest Model'''
    print(f'Training model with n_estimators={n_estimators}, max_depth={max_depth}...')
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    '''Evaluate model performance'''
    print('Evaluating model...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    print(f'\nAccuracy: {accuracy:.4f}')
    print(f'\nClassification Report:')
    print(report)

    return accuracy, report

def main():
    '''Main training Pipeline'''
    print('Starting MLOPs Training Pipeline')
    
    mlflow.set_experiment('iris-classification')

    with mlflow.start_run():
        n_estimators = 100
        max_depth = 3
        test_size = 0.3

        mlflow.log_param('n_estimators',n_estimators)
        mlflow.log_param('max_depth',max_depth)
        mlflow.log_param('test_size',test_size)

        X,y = load_data()
        X_train, X_test, y_train, y_test = prepare_data(X,y,test_size)

        model = train_model(X_train, y_train, n_estimators, max_depth)

        accuracy, classification_report = evaluate_model(model, X_test, y_test)

        mlflow.log_metric('accuracy',accuracy)

        mlflow.sklearn.log_model(model, 'random_forest_model')

        print('Trainig Complete! Model logged to MLflow.')

if __name__ == '__main__':
    main()