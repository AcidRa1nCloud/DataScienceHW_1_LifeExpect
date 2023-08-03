import pathlib
import logging
import torch
import numpy as np
from tqdm.auto import tqdm
from params import *
from utils import RunningAverage, load_checkpoint, save_checkpoint, save_to_json


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    summ_step: int=SUMMARY_SS,
    metrics: dict=METRICS,
    device: torch.device=DEVICE
    ):
    """
    Training part of training/validation loop

    Args:
        model: neural net
        dataloader: generator of batches (data, label)
        loss_fn: loss function to calculate the error between predicted and ground truth values
        optimizer: optimizer of model parameters
        summ_step: number of steps after which to store summary
        metrics: a dictionary of metrics we want to calculate for our models `accuracy`
        device: type of the device data is stored at
    """
    # Training mode
    model.train()

    # Make a summary list
    summary = []

    # Add running avarage object
    loss_avg = RunningAverage()

    # Create batch loop
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        logits = model(X)

        # Calculate loss
        loss = loss_fn(logits, y)

        # Clear gradients + backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Set new weights
        optimizer.step()

        # Updating summary list with metrics
        if i % summ_step == 0:
            logits = logits.data.cpu().numpy()
            y = y.data.cpu().numpy()
            # Loop and calculate metrics
            summ = {metric: metrics[metric](logits, y) for metric in metrics}
            summ['loss'] = loss.item()
            summary.append(summ)

        # Update average loss
        loss_avg.update(loss.item())


    # Compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summary]) for metric in summary[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info(" - Train metrics: " + metrics_string)
    return metrics_mean

def validation_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    metrics: dict=METRICS,
    device: torch.device=DEVICE
    ):
    """
    Validation part of training/validation loop

    Args:
        model: neural net
        dataloader: generator of batches (data, label)
        loss_fn: loss function to calculate the error between predicted and ground truth values
        metrics: a dictionary of metrics we want to calculate for our models `accuracy`
        device: type of the device data is stored at
    """
    # Evaluation mode
    model.eval()

    # Make a summary list
    summary = []

    # Add running avarage object
    loss_avg = RunningAverage()

    # Inference mode on
    with torch.inference_mode():
        # Create batch loop
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            logits = model(X)

            # Calculate loss
            loss = loss_fn(logits, y)

            # Update average loss
            loss_avg.update(loss.item())

            # Move tensors to np
            logits = logits.data.cpu().numpy()
            y = y.data.cpu().numpy()

            # Compute metrics
            summ = {metric: metrics[metric](logits, y) for metric in metrics}
            summ['loss'] = loss.item()
            summary.append(summ)

        # Compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric] for x in summary]) for metric in summary[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info(" - Eval metrics: " + metrics_string)

        return metrics_mean


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    summ_step: int=SUMMARY_SS,
    metrics: dict=METRICS,
    epochs: int=EPOCHS,
    restore_file: str=None,
    saving_path: str=SAVE_PATH,
    device: torch.device=DEVICE
    ):
    """
    Activation of training/validation loop

    Args:
        model: neural net
        train_dataloader: generator of training batches (data, label)
        valid_dataloader: generator of test batches (data, label)
        loss_fn: loss function to calculate the error between predicted and ground truth values
        optimizer: optimizer of model parameters
        summ_step: number of steps after which to store summary
        metrics: a dictionary of metrics we want to calculate for our models `accuracy`
        saving_path: a path were state dicts and json representation of them are saved
        device: type of the device data is stored at
    """

    # Reloade weight from restore file if specified
    if restore_file is not None:
        restore_file = restore_file + 'pth.tar'
        restore_path = SAVE_PATH / restore_file
        logging.info(f"Restored parameters from {restore_path}")
        load_checkpoint(restore_path, model, optimizer)

    # Set up summary variables
    best_val_acc = 0.0
    train_stats = []
    val_stats = []

    # Create train/test loop
    for epoch in tqdm(range(epochs)):
        # Print number of epochs
        logging.info(f"Epoch {epoch+1}/{epochs}")

        # Train
        train_metrics = train_step(model,
                                   train_dataloader,
                                   loss_fn,
                                   optimizer,
                                   summ_step,
                                   metrics,
                                   device)

        # Eval
        val_metrics = validation_step(model,
                                      valid_dataloader,
                                      loss_fn,
                                      metrics,
                                      device)

        # Save the best score of the chosen metric
        val_acc = val_metrics['MSE']
        is_best = val_acc >= best_val_acc

        # Save weights
        save_checkpoint({'epoch': epoch+1,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()},
                        is_best=is_best,
                        checkpoint=SAVE_PATH)

        # If best eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = SAVE_PATH / 'metrics_val_best_weights.json'
            save_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = SAVE_PATH / 'metrics_val_best_weights.json'
        save_to_json(val_metrics, best_json_path)

        train_stats.append(train_metrics)
        val_stats.append(val_metrics)

    return train_stats, val_stats
