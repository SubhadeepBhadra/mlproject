import os # the function of this module is to interact with the operating system, such as creating directories, joining paths, etc.
import sys # this module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter. It is always available.
from dataclasses import dataclass # this module provides a decorator and functions for automatically adding generated special methods such as __init__() and __repr__() to user-defined classes. It is available in Python 3.7 and later.

from sklearn.ensemble import ( # these are ensemble learning methods for regression tasks
    AdaBoostRegressor, # this is an ensemble learning method that combines multiple weak learners to create a strong learner for regression tasks.
    GradientBoostingRegressor, # this is an ensemble learning method that builds a model in a stage-wise fashion and generalizes them by allowing optimization of an arbitrary differentiable loss function for regression tasks.
    RandomForestRegressor, # this is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the mean prediction of the individual trees for regression tasks.
)
from sklearn.linear_model import LinearRegression # this is a linear model for regression tasks that assumes a linear relationship between the input variables and the target variable.
from sklearn.metrics import r2_score # this is a metric for evaluating the performance of regression models, which represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
from sklearn.neighbors import KNeighborsRegressor # this is a non-parametric method used for regression tasks that predicts the target variable based on the k-nearest neighbors in the feature space.
from sklearn.tree import DecisionTreeRegressor # this is a decision tree algorithm for regression tasks that splits the data into subsets based on the feature values and makes predictions based on the mean value of the target variable in each leaf node.
from xgboost import XGBRegressor # this is an optimized gradient boosting library that implements machine learning algorithms under the Gradient Boosting framework for regression tasks. It is designed to be highly efficient, flexible, and portable.

from src.exception import CustomException # this is a custom exception class defined in the src.exception module, which is used to handle exceptions in a specific way for this project.
from src.logger import logging # this is a logging module defined in the src.logger module, which is used to log messages for this project. It provides a way to configure and use logging in a consistent manner across the project.

from src.utils import save_object,evaluate_models # these are utility functions defined in the src.utils module. The save_object function is used to save a Python object to a file, and the evaluate_models function is used to evaluate multiple regression models and return their performance metrics.

@dataclass # this is a decorator that automatically generates special methods for the class, such as __init__() and __repr__(), based on the class attributes defined in the class body.
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found", sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)
            return r2
            



            
        except Exception as e:
            raise CustomException(e,sys)