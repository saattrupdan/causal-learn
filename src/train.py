'''Training script of GNN on sampled data'''

import torch
import torch.nn.functional as F
import torch.optim as opt
from torch_geometric.loader import DataLoader
import torchmetrics as tm
import itertools as it
from tqdm.auto import tqdm
from pathlib import Path
import json
import multiprocessing as mp

from data import CPDAGDataset
from model import CausalDiscoverer
from config import Config


def train(config: Config):
    '''Training script of GNN on sampled data'''

    # Initialise the model path
    model_dir = Path(config.model_dir)

    # Create the model directory if it doesn't exist
    if not model_dir.exists():
        model_dir.mkdir()

    # Set the model and config paths
    model_path = model_dir / 'model.pt'
    config_path = model_dir / 'config.json'

    # Load the data and set up a dataloader
    dataset = CPDAGDataset(config)
    dataloader = DataLoader(dataset, num_workers=mp.cpu_count())

    # Initialise the model
    model = CausalDiscoverer(config)

    # Move the model to the GPU if it is available
    if torch.cuda.is_available():
        model.cuda()

    # Load the model weights if they exist
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))

    # Set the model to training mode
    model.train()

    # Count the number of parameters in the model and print it
    num_params = sum(p.numel() for p in model.parameters())
    print(f'The model has {num_params:,} parameters.')

    # Report which device the model is on
    model_on_gpu = next(model.parameters()).is_cuda
    if model_on_gpu:
        print('The model lies on the GPU.')
    else:
        print('The model lies on the CPU.')

    # Initialise the metrics
    f1_metric = tm.F1(threshold=config.threshold)
    precision_metric = tm.Precision(threshold=config.threshold)
    recall_metric = tm.Recall(threshold=config.threshold)
    specificity_metric = tm.Specificity(threshold=config.threshold)

    # Move metrics to the GPU if it is available
    if torch.cuda.is_available():
        f1_metric.cuda()
        precision_metric.cuda()
        recall_metric.cuda()
        specificity_metric.cuda()

    # Define the optimiser
    optimiser = opt.AdamW(model.parameters(), lr=config.lr)

    # Initialise the exponentially moving average of the loss and metrics
    ema_loss = 1.
    ema_f1 = 0.
    ema_precision = 0.
    ema_recall = 0.
    ema_specificity = 0.

    # Set up a progress bar
    with tqdm(enumerate(it.islice(dataloader, config.num_iterations)),
               total=config.num_iterations,
               desc='Training') as pbar:

        # Train the model
        for iter_idx, batch in pbar:
            data, y = batch

            # Move the data to the GPU if it is available
            if torch.cuda.is_available():
                data = data.cuda()
                y = y.cuda()

            # Forward pass
            edge_probabilities = model(data)

            # Calculate the loss
            loss = F.binary_cross_entropy(edge_probabilities, y.squeeze(0))

            # Compute the metrics
            f1 = f1_metric(edge_probabilities, y.squeeze(0).int())
            precision = precision_metric(edge_probabilities,
                                         y.squeeze(0).int())
            recall = recall_metric(edge_probabilities, y.squeeze(0).int())
            specificity = specificity_metric(edge_probabilities,
                                             y.squeeze(0).int())

            # Calculate the exponential moving average of the loss
            ema_loss = (config.ema_decay * ema_loss +
                        (1 - config.ema_decay) * float(loss.item()))

            # Calculate the exponential moving average of the f1 score
            ema_f1 = (config.ema_decay * ema_f1 +
                      (1 - config.ema_decay) * float(f1))

            # Calculate the exponential moving average of the precision
            ema_precision = (config.ema_decay * ema_precision +
                             (1 - config.ema_decay) * float(precision))

            # Calculate the exponential moving average of the recall
            ema_recall = (config.ema_decay * ema_recall +
                          (1 - config.ema_decay) * float(recall))

            # Calculate the exponential moving average of the specificity
            ema_specificity = (config.ema_decay * ema_specificity +
                               (1 - config.ema_decay) * float(specificity))

            # Backpropagate the loss
            loss.backward()

            # Update gradients every `config.batch_size` iterations
            if (iter_idx + 1) % config.batch_size == 0:

                # Update the optimiser
                optimiser.step()

                # Zero the gradients
                optimiser.zero_grad()

                # Update the progress bar
                pbar.set_description(
                    f'Training - loss {ema_loss:.4f} - '
                    f'F1 score {100 * ema_f1:.2f} - '
                    f'Precision {100 * ema_precision:.2f} - '
                    f'Recall {100 * ema_recall:.2f} - '
                    f'Specificity {100 * ema_specificity:.2f}'
                )

    # Save the config
    with config_path.open('w') as f:
        json.dump(config.__dict__, f)

    # Save the model
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    config = Config()
    train(config)
