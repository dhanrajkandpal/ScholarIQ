import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig,DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "raw.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion")

        try:
            # Read dataset
            ROOT_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )

            DATA_PATH = os.path.join(ROOT_DIR, "src", "stud.csv")

            df = pd.read_csv(DATA_PATH)

            logging.info("Dataset loaded successfully")

            # Create artifact directory
            artifact_dir = os.path.join(ROOT_DIR, "artifact")
            os.makedirs(artifact_dir, exist_ok=True)

            # Save raw dataset
            raw_path = os.path.join(ROOT_DIR, self.ingestion_config.raw_data_path)

            df.to_csv(raw_path, index=False, header=True)

            logging.info("Train test split initiated")

            # Split dataset
            train_set, test_set = train_test_split(
                df,
                test_size=0.25,
                random_state=42
            )

            # Save training dataset
            train_path = os.path.join(
                ROOT_DIR, self.ingestion_config.train_data_path
            )

            train_set.to_csv(train_path, index=False, header=True)

            # Save testing dataset
            test_path = os.path.join(
                ROOT_DIR, self.ingestion_config.test_data_path
            )

            test_set.to_csv(test_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                train_path,
                test_path,
                raw_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataIngestion()
    train_data, test_data, _ = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)