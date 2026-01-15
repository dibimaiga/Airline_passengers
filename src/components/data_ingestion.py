import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "raw_train.csv")
    test_data_path: str = os.path.join("artifacts", "raw_test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started.")
        try:
            # Chemins source (CSV Kaggle)
            train_src = os.path.join("notebook", "data", "train.csv")
            test_src = os.path.join("notebook", "data", "test.csv")

            if not os.path.exists(train_src):
                raise FileNotFoundError(f"train.csv introuvable: {train_src}")
            if not os.path.exists(test_src):
                raise FileNotFoundError(f"test.csv introuvable: {test_src}")

            # Lecture
            df_train = pd.read_csv(train_src)
            df_test = pd.read_csv(test_src)

            logging.info(f"Train shape: {df_train.shape}; Test shape: {df_test.shape}")

            # Création dossier artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Sauvegarde copies raw
            df_train.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("Raw train/test saved into artifacts folder.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.exception("Error during data ingestion.")
            raise CustomException(e, sys)


if __name__ == "__main__": #This code only runs when you execute this file directly
# Allows you to test this component standalone
# Doesn't run when imported by other files
# Common Python pattern

# Exemple: # Importing from another file:
# from src.components.data_ingestion import DataIngestion  # ✗ Code inside doesn't run

    obj = DataIngestion() #initialization(data ingestion object)
    # train_data,test_data = 
    obj.initiate_data_ingestion()

# #So abve we've combined dataingestion then down we've combined DataTransformation
    
    # data_transformation = DataTransformation()  # To initialize (and you'll see that it will be able to call this self.data_transformation_config function)
    # train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

# #the third i don't need it cause i've already created the pkl file

# model_trainer = ModelTrainer()

# print(model_trainer.initiate_model_trainer(train_arr,test_arr))


