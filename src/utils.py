import os # this is for creating directory and file handling
import sys # this is for exception handling and to get the details of the exception

import numpy as np # this is for numerical operations
import pandas as pd # this is for data manipulation and analysis
import dill # this is for serializing and deserializing Python objects, similar to pickle but with more features
import pickle # this is for serializing and deserializing Python objects, it converts Python objects into a byte stream and vice versa
from sklearn.metrics import r2_score # this is for evaluating the performance of regression models, it calculates the R-squared score which indicates how well the model fits the data
from sklearn.model_selection import GridSearchCV # this is for hyperparameter tuning, it performs an exhaustive search over specified parameter values for an estimator and finds the best combination of parameters that gives the best performance on the training data

from src.exception import CustomException # this is for handling custom exceptions, it allows us to create our own exception class that can provide more specific error messages and handle exceptions in a more controlled way

def save_object(file_path, obj): # this function is for saving a Python object to a file, it takes the file path and the object to be saved as arguments
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))): 
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            model_name = list(models.keys())[i]

            try:
                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_) 
                model.fit(X_train,y_train)
            except AttributeError as ae:
                # Skip sklearn incompatible models (e.g., CatBoost with newer sklearn)
                if '__sklearn_tags__' in str(ae):
                    print(f"Skipping {model_name} due to sklearn compatibility issue")
                    continue
                raise

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)