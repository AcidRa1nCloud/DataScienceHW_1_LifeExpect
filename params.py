import os
import torch
import pathlib
from sklearn.metrics import mean_squared_error

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
LR = 0.01
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else None
SAVE_PATH = pathlib.Path('./saved_models')
SUMMARY_SS = 1
LOSS_FN = torch.nn.MSELoss()
OPTIM = torch.optim.Adam
METRICS = {'MSE': mean_squared_error}

if not SAVE_PATH.is_dir():
    print('Creating a directory for saving models')
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
