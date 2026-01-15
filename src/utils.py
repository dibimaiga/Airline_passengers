#Utils we'll have all the common things

import os
import sys

import numpy as np 


import pandas as pd
import pickle # this will help us create the pkl file
# pickle: It converts Python objects
# (like trained models, scalers, encoders) into a byte stream that can be saved to disk,
# then loaded back later. Think of it as "freezing" your object to use it later.
# Pickle lets you separate training time (expensive, one-time)
# from inference time (cheap, repeated). Without it, you couldn't build practical ML applications.

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import roc_auc_score, f1_score

def save_object(file_path, obj):
    """
    file_path: A string like "artifacts/model.pkl" - where to save the file

    obj: The Python object you want to save (could be a model, preprocessor, dictionary, etc.)
    """
    try:
        dir_path = os.path.dirname(file_path)
#         os.path.dirname(): Extracts the directory path from the full file path.
# Example: If file_path = "artifacts/models/model.pkl", then dir_path = "artifacts/models"
# This is needed because we need to ensure the directory exists before saving the file.

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
# "with" = Automatically manage opening/closing
# "open(...)" = Open the file
# "model.pkl" = File path
# "wb" = Write mode + Binary mode
# "as file_obj" = Call it 'file_obj' so I can use it
# pickle.dump(...) = Save the model to that file
# (exit block) = File automatically closes

    except Exception as e: #Catches any exception that occurred in the try block and assigns it to variable e.
        raise CustomException(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def zero_to_nan(X):
    # X arrives as a numpy array (via ColumnTransformer), we convert it properly
    X = np.array(X, dtype=float, copy=True)
    X[X == 0] = np.nan
    return X


# def evaluate_models(
#     X_train, y_train, X_test, y_test,
#     models: dict, 
#     scoring: str = "roc_auc",
#     cv_splits: int = 3,
#     random_state: int = 42
# ):
#     """
#     Retourne:
#       - report: dict[model_name] = score sur le test (roc_auc ou f1)
#       - best_estimators: dict[model_name] = estimator optimisé (best_estimator_)

#     scoring:
#       - "roc_auc" => basé sur predict_proba
#       - "f1" => basé sur predict (seuil 0.5 si proba)
#     """
#     try:
#         report = {}
#         best_estimators = {}

#         cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

#         for model_name, model in models.items():
#             logging.info(f"GridSearchCV start: {model_name}")

#             param_grid = params.get(model_name, {})

#             gs = GridSearchCV(
#                 estimator=model,
#                 param_grid=param_grid,
#                 cv=cv,
#                 scoring=scoring,
#                 n_jobs=-1,
#                 refit=True
#             )

#             gs.fit(X_train, y_train)
#             best_model = gs.best_estimator_
#             best_estimators[model_name] = best_model

#             # --- score sur test ---
#             # if scoring == "roc_auc":
#             if not hasattr(best_model, "predict_proba"):
#                 report[model_name] = np.nan
#                 logging.info(f"{model_name} has no predict_proba => roc_auc=NaN")
#             else:
#                 proba = best_model.predict_proba(X_test)[:, 1]
#                 report[model_name] = roc_auc_score(y_test, proba)


#             logging.info(f"{model_name} best_params={gs.best_params_} test_{scoring}={report[model_name]}")

#         return report, best_estimators

#     except Exception as e:
#         raise CustomException(e, sys)
