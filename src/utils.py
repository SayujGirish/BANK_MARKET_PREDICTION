import os
import sys
import dill

from src.exception import CustomException
import logging
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Evaluation Metrics
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            grid=GridSearchCV(model,para,cv=3)

            # Train model
            grid.fit(X_train, y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)