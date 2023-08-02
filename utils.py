import pathlib
import logging
import json
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display

class RunningAverage():
    """
    Running average object calculates the average of values passed on each step,
    returns the average when called
    """

    def __init__(self):
        self.total = 0
        self.steps = 0

    def update(self, val):
        self.total+=val 
        self.steps+=1

    def __call__(self):
        return self.total/float(self.steps)


def train_val_split(
    X: np.array, 
    y: np.array, 
    train_split: int=70
    ):
    """
    Acts like train_test_split() from sklearn + validation split

    Args:
        X: Feature matrix
        y: Target vector
        train_split: percent of train split

    Returns:
        Like sklearn function returns slices into train and test data
    """
    val_split = (100 - train_split) // 2
    t = len(y)
    p_s = t*train_split//100
    p_f = (t*val_split//100)+p_s
    X_train, y_train, X_val, y_val, X_test, y_test = X[:p_s], y[:p_s], X[p_s:p_f], y[p_s:p_f], X[p_f:], y[p_f:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def set_logger(
    log_path: str
    ):
    """
    Outputs logs of the process into terminal bar + saves them into given path
    as a text file

    Args:
        log_path: path to the log file 
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File logging
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # terminal loggging
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(
    state: dict, 
    is_best: bool, 
    checkpoint: str
    ):
    """
    Saves model and training parameter at checkpoints + `last.pth.tar`,
    has an option to save the best result `is_best==True` adds `best.pth.tar`

    Args:
        state: models state dict, may contain hyper parameter values
        is_best: True if it is the best model so far
        checkpoint: path to the folder with saved state dicts
    """
    dir_path = pathlib.Path(checkpoint)
    filepath = dir_path / 'last.pth.tar'
    if not dir_path.is_dir():
        print(f'Directory don\'t exist! Creating directory {checkpoint}.')
        dir_path.mkdir(parents=True, exist_ok=True)
    print('Saving checkpoint!')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, dir_path / 'best.pth.tar')


def load_checkpoint(
    checkpoint: str, 
    model: torch.nn.Module,
    optimizer: torch.optim=None
    ):
    """
    Loads models state dict from filepath,
    if optimizer provided => loads state_dict of optimizer

    Args:
        checkpoint: filename to load
        model: instance of a model where to load the state dict
        optimizer: if set adds an optimizer with tunned hyper params
    """
    dir_path = pathlib.Path(checkpoint)
    if not dir_path.isdir():
        raise (f'Directory {checkpoint} don\'t exist')
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_to_json(
    dictionary: dict, 
    json_path: str
    ):
    """
    Parses a dict to a json file and saves it to a given path

    Args:
        dictionary: a dictionary
        json_path: path name where to save the file
    """
    path = pathlib.Path(json_path)
    with open(path, 'w') as file:
        dictionary = {k: float(v) for k, v in dictionary.items()}
        json.dump(dictionary, file, indent=4)

def plot_results(
    train_stats: np.array, 
    val_stats: np.array
    ):
    """
    Plots the loss curve and MSE curve

    Args:
        train_stats: train metrics
        val_stats: val metrics
    """
    # List of dicts to dict of lists
    train_stats = {k: [dic[k] for dic in train_stats] for k in train_stats[0]}
    val_stats = {k: [dic[k] for dic in val_stats] for k in val_stats[0]}

    # Set subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    # Plot first subplot
    ax1.plot(train_stats['loss'])
    ax1.plot(val_stats['loss'])

    # Set first title and legend
    ax1.set_title('Loss')
    ax1.legend(['Train', 'Validation'])

    # Plot second subplot
    ax2.plot(train_stats['MSE'])
    ax2.plot(val_stats['MSE'])

    # Set second title and legend
    ax2.set_title('MSE')
    ax2.legend(['Train', 'Validation'], loc='lower left')

    # Save image
    plt.savefig('plot.png')

    # Show image
    display(Image(filename='plot.png'))
