import logging
import os
from datetime import datetime

log_file = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

logs_path = os.path.join(os.getcwd(), "logs", log_file)

os.makedirs(os.path.dirname(logs_path), exist_ok=True)

log_file_path = logs_path

logging.basicConfig(
filename=log_file_path,
format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
level=logging.INFO
)

