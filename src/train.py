'''Training script of GNN on sampled data'''

import torch.nn.functional as F
import torch.optim as opt
from torch_geometric.loader import DataLoader
import itertools as it
from tqdm.auto import tqdm
from data import CorrDataset
from model import CausalDiscoverer
from config import Config


def train(config: Config):
    '''Training script of GNN on sampled data'''

    # Load the data and set up a dataloader
    dataset = CorrDataset()
    dataloader = DataLoader(dataset)

    # Initialise the model
    model = CausalDiscoverer(dim=config.dim, dropout=config.dropout)
    model.train()

    # Define the optimiser
    optimiser = opt.AdamW(model.parameters(), lr=config.lr)

    # Train the model
    ema_loss = 0.
    with tqdm(enumerate(it.islice(dataloader, config.num_iterations)),
               total=config.num_iterations,
               desc='Training') as pbar:
        for iter_idx, batch in pbar:
            data, y = batch
            edge_probabilities = model(data)
            loss = F.binary_cross_entropy(edge_probabilities, y.squeeze(0))
            loss.backward()

            # Update gradients every `config.batch_size` iterations
            if (iter_idx + 1) % config.batch_size == 0:

                optimiser.step()
                optimiser.zero_grad()

                # Calculate the exponential moving average of the loss
                ema_loss = (config.ema_decay * ema_loss +
                            (1 - config.ema_decay) * loss.item())
                pbar.set_description(f'Training... loss = {ema_loss:.4f}')

if __name__ == '__main__':
    config = Config()
    train(config)
