import sys
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info(f"Splitting train and test set")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CATBoost": CatBoostRegressor(),
            }
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", 1.0]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [2, 3, 5],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.8, 0.9, 1.0]
                },

                "AdaBoost": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                    "loss": ["linear", "square", "exponential"]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"]
                },

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7, 9, 11, 15],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size": [20, 30, 40],
                    "p": [1, 2]
                },

                "XGBoost": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0],
                    "min_child_weight": [1, 3, 5],
                    "gamma": [0, 0.1, 0.2]
                },

                "CATBoost": {
                    "iterations": [500, 800, 1000],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "l2_leaf_reg": [1, 3, 5, 7],
                    "verbose": [False]
                }
            }
            model_report:dict = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                param = params,
            )
            #to get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            #to get the best model from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score<.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found")
            save_object(
                file_path=os.path.join(self.model_trainer_config.trained_model_file_path),
                obj=best_model,
                        )
            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test,predicted)
            print(f"\nBest Model: {best_model_name}")
            print(f"Best Model R2 Score: {r2:.4f}")
            return r2
        except Exception as e:
            raise CustomException(e,sys)