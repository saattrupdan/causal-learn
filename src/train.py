'''Training script of GNN on sampled data'''

import torch
import torch.nn.functional as F
import torch.optim as opt
from torch_geometric.loader import DataLoader
import torchmetrics as tm
import itertools as it
from tqdm.auto import tqdm
from pathlib import Path

from data import CorrDataset
from model import CausalDiscoverer
from config import Config


def train(config: Config):
    '''Training script of GNN on sampled data'''

    # Initialise the model path
    model_path = Path(config.model_path)

    # Load the data and set up a dataloader
    dataset = CorrDataset()
    dataloader = DataLoader(dataset)

    # Initialise the model
    model = CausalDiscoverer(dim=config.dim, dropout=config.dropout)

    # Initialise the F1 metric
    f1_metric = tm.F1(threshold=config.threshold)

    # Load the model weights if they exist
    if model_path.exists():
        model.load_state_dict(torch.load(config.model_path))

    # Set the model to training mode
    model.train()

    # Define the optimiser
    optimiser = opt.AdamW(model.parameters(), lr=config.lr)

    # Initialise the exponentially moving average of the loss and f1 score
    ema_loss = 1.
    ema_f1 = 0.

    # Set up a progress bar
    with tqdm(enumerate(it.islice(dataloader, config.num_iterations)),
               total=config.num_iterations,
               desc='Training') as pbar:

        # Train the model
        for iter_idx, batch in pbar:
            data, y = batch

            # Forward pass
            edge_probabilities = model(data)

            # Calculate the loss
            loss = F.binary_cross_entropy(edge_probabilities, y.squeeze(0))

            # Compute the f1 score
            f1 = f1_metric(edge_probabilities, y.squeeze(0).int())

            # Calculate the exponential moving average of the loss
            ema_loss = (config.ema_decay * ema_loss +
                        (1 - config.ema_decay) * float(loss.item()))

            # Calculate the exponential moving average of the f1 score
            ema_f1 = (config.ema_decay * ema_f1 +
                      (1 - config.ema_decay) * float(f1))

            # Backpropagate the loss
            loss.backward()

            # Update gradients every `config.batch_size` iterations
            if (iter_idx + 1) % config.batch_size == 0:

                # Update the optimiser
                optimiser.step()

                # Zero the gradients
                optimiser.zero_grad()

                # Update the progress bar
                pbar.set_description(f'Training - loss {ema_loss:.4f} - '
                                     f'F1 score {100 * ema_f1:.2f}')

    # Save the model
    torch.save(model.state_dict(), config.model_path)

if __name__ == '__main__':
    config = Config()
    train(config)
