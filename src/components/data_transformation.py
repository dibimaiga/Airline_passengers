#Feature engineering, data cleaning, convert categoorical features into numerical ones

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,zero_to_nan


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def _drop_technical_cols(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["Unnamed: 0", "id"], errors="ignore")

    @staticmethod
    def _encode_target(y: pd.Series) -> np.ndarray:
        # 1 = satisfied ; 0 = neutral or dissatisfied
        return (y.astype(str).str.strip().str.lower() == "satisfied").astype(int).values

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        This function is reponsible for data transformation

        """
        try:
            ratings_columns = ["Inflight wifi service","Departure/Arrival time convenient",
            "Ease of Online booking","Gate location","Food and drink","Online boarding",
            "Seat comfort","Inflight entertainment","On-board service","Leg room service",
            "Baggage handling","Checkin service","Inflight service","Cleanliness"]

            num_cols = ["Age","Flight Distance","Arrival Delay in Minutes",]

            cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]

            logging.info(f"Categorical cols: {cat_cols}")
            logging.info(f"Numeric cols: {num_cols}")
            logging.info(f"Rating cols: {ratings_columns}")

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            rating_pipeline = Pipeline(steps=[
                    ("zero_to_nan", FunctionTransformer(zero_to_nan, feature_names_out="one-to-one")),
                    # add_indicator=True => ajoute des colonnes binaires "_missing"
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ])

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            logging.info("Numerical columns Standard scaling completed")

            logging.info("Ratings columns zero_to_nan completed")

            logging.info("Categorical columns Standard scaling completed")


            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, num_cols),
                    ("rating", rating_pipeline, ratings_columns),
                    ("cat", categorical_pipeline, cat_cols),
                ],
                remainder="drop",
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info("Data transformation started.")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read test and train data completed")

            train_df = self._drop_technical_cols(train_df)
            test_df = self._drop_technical_cols(test_df)

            target_col = "satisfaction"
            if target_col not in train_df.columns or target_col not in test_df.columns:
                raise ValueError("The 'satisfaction' column must be present in train and test.")

            # split X/y
            y_train = self._encode_target(train_df[target_col])
            y_test = self._encode_target(test_df[target_col])

            X_train = train_df.drop(columns=[target_col])
            X_test = test_df.drop(columns=[target_col])

            # sanity schema
            if set(X_train.columns) != set(X_test.columns):
                missing_in_test = sorted(list(set(X_train.columns) - set(X_test.columns)))
                missing_in_train = sorted(list(set(X_test.columns) - set(X_train.columns)))
                raise ValueError(
                    f"Different feature diagram between train and test. "
                    f"Missing in test: {missing_in_test} | Missing in train: {missing_in_train}"
                )
            
            logging.info("Obtaining prepocessing object")

            preprocessor = self.get_data_transformer_object()

            logging.info(f"Applying prepocessing object on test and train dataframes ")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed, y_test]

            os.makedirs("artifacts", exist_ok=True)

            logging.info("Saved prepocessing object  ")


            save_object(self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessor)

            logging.info("Preprocessor + transformed arrays saved in artifacts.")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.exception("Error during data transformation.")
            raise CustomException(e, sys)