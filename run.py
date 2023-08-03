import warnings
import argparse
import pathlib
import logging
import torch
from torch.utils.data import DataLoader
from params import *
from utils import *
from models import *
from data import train_dataset, valid_dataset, test_dataset
from engine import train

# Parse model arguments
parser = argparse.ArgumentParser(description='A parser that can recieve hyper parameters for a model NN to train on')
parser.add_argument('-e','--epochs', help='Number of epochs to run through', default=100, type=int)
parser.add_argument('--lr', help='Set learning rate of optimizer', default=0.001, type=float)
parser.add_argument('-b','--batch', help='Set batch size of dataloader', default=32, type=int)
parser.add_argument('--in_units', help='Set in units', default=21, type=int)
parser.add_argument('--hidden_units', help='Set hidden units', default=32, type=int)
parser.add_argument('--out_units', help='Set out units', default=1, type=int)
parser.add_argument('--sstep', help='Set summary step', default=1, type=int)
parser.add_argument('-d', '--dropout', help='Set dropout probability', default=0.5, type=float)
parser.add_argument('-m', '--model', help='Choose the model', default='V0', type=str, choices=['V0', 'V1', 'V2'])
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch
SUMMARY_SS = args.sstep

# Shut down warnings
warnings.filterwarnings('ignore')

# Set random seed to 42
torch.manual_seed(42)
if DEVICE=='cuda':
    torch.cuda.manual_seed(42)

# Set logger
set_logger(SAVE_PATH / 'train.log')

# Instanciate model
logging.info("Model instance")
model_v = model_select(args.model)
model = model_v(in_units=args.in_units, hidden_units=args.hidden_units, out_units=args.out_units, dropout_prob=args.dropout).to(DEVICE)

# Set loss function and optimizer
logging.info("Set loss function")
loss_fn = LOSS_FN
logging.info("Set optimizer")
optimizer = OPTIM(params=model.parameters(), lr=args.lr)

# Create dataloaders
logging.info("Loading the datasets...")
train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

valid_dataloader = DataLoader(valid_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=False)

logging.info("- all done.")

# Train the model
logging.info(f"Start training for {EPOCHS} epoch(s)")
train_stats, val_stats = train(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    summ_step=SUMMARY_SS,
    metrics=METRICS,
    epochs=EPOCHS,
    restore_file=None,
    saving_path=SAVE_PATH,
    device=DEVICE)

plot_results(train_stats, val_stats)
