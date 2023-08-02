import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import train_val_split

array_dir = pathlib.Path('./data')
if not array_dir.is_dir():
    raise Exception("There is no such directory! Create one.")

X, y = np.load(array_dir / 'X.npy'), np.load(array_dir / 'y.npy')
X_train, y_train, X_val, y_val, X_test, y_test = train_val_split(X, y)

class Life_Expect_Dataset(Dataset):
    """
    Custom Dataset for Life expectancy csv data
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y

train_dataset = Life_Expect_Dataset(X_train, y_train)
valid_dataset = Life_Expect_Dataset(X_val, y_val)
test_dataset = Life_Expect_Dataset(X_test, y_test)
