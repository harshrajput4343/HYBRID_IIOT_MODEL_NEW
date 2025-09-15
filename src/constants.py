# src/constants.py

import os 
SEED = 42
DATA_PATH = os.path.join("data", "train_test_network.csv")

TARGET_COL = "type"        # <-- use attack type as target
TIMESTAMP_COL = "ts"       # keep if you plan time-aware splits
