import os
import sys
from dataclasses import dataclass

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
	trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
	def __init__(self):
		self.model_trainer_config = ModelTrainerConfig()

	def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray):
		try:
			logging.info("Splitting training and test input arrays")

			X_train, y_train = train_array[:, :-1], train_array[:, -1]
			X_test, y_test = test_array[:, :-1], test_array[:, -1]

			models = {
				"RandomForest": RandomForestRegressor(),
				"GradientBoosting": GradientBoostingRegressor(),
				"LinearRegression": LinearRegression(),
			}

			params = {
				"RandomForest": {"n_estimators": [50], "max_depth": [None]},
				"GradientBoosting": {"n_estimators": [50], "learning_rate": [0.1]},
				"LinearRegression": {},
			}

			model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

			best_model_name = max(model_report, key=model_report.get)
			best_model_score = model_report[best_model_name]
			best_model = models[best_model_name]

			save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

			return best_model_score

		except Exception as e:
			raise CustomException(e, sys)

