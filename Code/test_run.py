import os
import sys
from datetime import datetime as dtime
import logging


## 1. Setup
now = dtime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"/home/sranasin/StatisticalMachineLearning/Logs/log_{now}.log"
logging.basicConfig(
    filename=log_path,
    filemode="w",  # Overwrite each time; use "a" to append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Logging Started")
