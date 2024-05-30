import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    fig, axs = plt.subplots(1, 1, figsize=(13, 4))
    axs.plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs.plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs.set_ylabel('loss')

    axs.set_xlabel('epoch')
    axs.legend()
    plt.grid()
    plt.show()


def training_epoch(model, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: RNN to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0
    model.train()
    for X_batch, y_batch in tqdm(loader, desc=tqdm_desc):
        optimizer.zero_grad()

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        output = model.forward(X_batch)

        loss = criterion(output.flatten(), y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= len(loader.dataset)
    return train_loss.cpu().detach()


@torch.no_grad()
def validation_epoch(model, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for X_batch, y_batch in tqdm(loader, desc=tqdm_desc):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        output = model.forward(X_batch)
        loss = criterion(output.flatten(), y_batch)
        val_loss += loss

    val_loss /= len(loader.dataset)
    return val_loss.cpu()


def train(model,
          optimizer: torch.optim.Optimizer,
          train_loader: DataLoader,
          val_loader: DataLoader,
          num_epochs: int):
    """
    Train RNN for several epochs
    :param model: rnn to train
    :param optimizer: optimizer instance
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    """

    train_losses, val_losses = [], []
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        train_losses += [train_loss]
        val_losses += [val_loss]
        plot_losses(train_losses, val_losses)
