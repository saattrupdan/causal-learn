'''GNN model used to predict CPDAG from a correlation matrix'''

import torch_geometric.nn as tgnn
from torch_geometric.data import Data
from torch_geometric.utils.dropout import dropout_adj
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Union

from config import Config


class CausalDiscoverer(nn.Module):
    '''GNN model used to predict CPDAG from a correlation matrix.

    Args:
        config (Config):
            Configuration object containing hyperparameters for the model.

    Attributes:
        dim (int):
            The node dimension.
        dropout (float):
            The dropout rate.
        conv1 (torch.nn.Conv1d):
            The first convolutional layer.
        conv2 (torch.nn.Conv1d):
            The second convolutional layer.
        edge_mlp (torch.nn.Sequential):
            The MLP used to get the edge probabilities.
    '''
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.dropout = config.dropout
        self.conv1 = tgnn.GINConv(
            nn.Sequential(nn.Linear(config.num_data_points, config.dim),
                          nn.LayerNorm(config.dim),
                          nn.GELU(),
                          nn.Dropout(config.dropout),
                          nn.Linear(config.dim, config.dim))
        )
        self.conv2 = tgnn.GINConv(
            nn.Sequential(nn.Linear(config.dim, config.dim),
                          nn.LayerNorm(config.dim),
                          nn.GELU(),
                          nn.Dropout(config.dropout),
                          nn.Linear(config.dim, config.dim))
        )
        self.edge_mlp = nn.Sequential(nn.Linear(config.dim * 2, config.dim),
                                      nn.LayerNorm(config.dim),
                                      nn.GELU(),
                                      nn.Dropout(config.dropout),
                                      nn.Linear(config.dim, 1),
                                      nn.Sigmoid())

        # Initialise a random generator
        self._rng = np.random.default_rng()

    def forward(self, data: Data) -> torch.Tensor:
        '''Get node features from a correlation/causal graph.

        Args:
            data (PyG Data object):
                The graph data.

        Returns:
            PyTorch tensor:
                The edge probabilities
        '''
        # Extract the graph data
        x, edge_index = data.x, data.edge_index

        # Drop causal edges if training
        if self.training:

            # Sample causal dropout rate from Unif[0, 1], using `self._rng`
            causal_dropout = self._rng.uniform(0, 1)

            # Dropout causal edges with probability `causal_dropout`
            edge_index, _ = dropout_adj(edge_index=edge_index,
                                        p=causal_dropout,
                                        training=self.training)
            edge_index = edge_index

        # Apply convolutional layers to get the node features
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        # Extract edge features from the node features. This concatenates every
        # pair of node features in `x`, ending up with a tensor of shape
        # (num_nodes, num_nodes, `self.dim` * 2)
        num_nodes = x.size(0)
        edge_feats = torch.zeros((num_nodes, num_nodes, self.dim * 2))
        edge_feats[:, :, :self.dim] = (x.repeat(1, num_nodes)
                                        .view(num_nodes, num_nodes, self.dim))
        edge_feats[:, :, self.dim:] = (x.repeat(1, num_nodes)
                                        .view(num_nodes, num_nodes, self.dim)
                                        .transpose(0, 1))

        # Apply the MLP to get the edge probabilities, ending up with a tensor
        # of shape (num_nodes, num_nodes)
        edge_probs = self.edge_mlp(edge_feats).squeeze(-1)

        # Return the edge probabilities
        return edge_probs

    def predict(self,
                data_matrix: Union[np.ndarray, pd.DataFrame],
                threshold: float = 0.5) -> np.ndarray:
        '''Predict CPDAG from a data matrix.

        This CPDAG is computed by successively applying the `forward` method on
        the correlation graph, adding a causal edge at each step.

        Args:
            data_matrix (NumPy array or Pandas DataFrame):
                A data matrix, of shape (n_samples, n_features).
            threshold (float):
                Probability threshold to include a causal edge.

        Returns:
            NumPy array:
                The adjacency matrix of a CPDAG, of shape
                (n_features, n_features).
        '''
        # Disable gradient accumulation
        with torch.no_grad():

            # Ensure that the model is in evaluation mode, which for instance
            # disables the dropout layers.
            self.eval()

            # Store the number of features
            num_feats = data_matrix.shape[1]

            # Compute the correlation matrix of `data_matrix`
            corr_matrix = np.corrcoef(data_matrix.T)

            # Organise the input as a PyG graph, to be inputted to the model
            graph_data = HeteroData()
            graph_data['feat'].x = torch.tensor(data_matrix)
            graph_data['feat', 'correlated_with', 'feat'].edge_index = \
                torch.ones((2, num_feats * num_feats)).long()
            graph_data['feat', 'correlated_with', 'feat'].edge_attr = \
                (torch.tensor(corr_matrix)
                      .view(num_feats * num_feats, 1)
                      .float())
            graph_data['feat', 'implies', 'feat'].edge_index = \
                torch.empty(2, 0, dtype=torch.long)

            # Loop until no more causal edges are added
            while True:

                # Get the edge probabilities
                edge_probs = self.forward(graph_data)

                # Get the indices of the edges that we have already added
                eidx = graph_data['feat', 'implies', 'feat'].edge_index
                added_edges = [(int(i), int(j))
                               for i, j in zip(eidx[0], eidx[1])]

                # Get the maximum edge probability among the edges that we
                # have not already added
                max_prob = max([edge_probs[i, j]
                                for i in range(num_feats)
                                for j in range(num_feats)
                                if (i, j) not in added_edges])

                # If all probabilities are below the threshold then halt
                if max_prob < threshold:
                    break

                # Otherwise, add a causal edge with the highest probability
                else:
                    # Get the node pair whose edge probability is largest,
                    # among the edges not already added as causal edges
                    src, tgt = [(i, j)
                                for i in range(num_feats)
                                for j in range(num_feats)
                                if (i, j) not in added_edges
                                and edge_probs[i, j] == max_prob][0]

                    # Get the existing causal edge indices
                    causal_edge_index = (graph_data['feat', 'implies', 'feat']
                                         .edge_index)

                    # Add the new causal edge
                    new_edge = torch.tensor([[src], [tgt]])
                    causal_edge_index = torch.cat((causal_edge_index,
                                                   new_edge),
                                                  dim=1)
                    graph_data['feat', 'implies', 'feat'].edge_index = \
                        causal_edge_index

        # Get the causal edges
        causal_edges = graph_data['feat', 'implies', 'feat'].edge_index.numpy()

        # Convert the causal edges to an adjacency matrix
        adj_matrix = np.zeros((num_feats, num_feats))
        adj_matrix[causal_edges[0], causal_edges[1]] = 1

        # Return the adjacency matrix of the predicted CPDAG
        return adj_matrix


if __name__ == '__main__':
    from data import CorrDataset
    from torch_geometric.loader import DataLoader

    # Load the data and set up a dataloader
    dataset = CorrDataset()
    dataloader = DataLoader(dataset)

    # Initialise the model
    model = CausalDiscoverer(dim=16, dropout=0.5)

    # Pass some data through the model
    for batch in dataloader:
        data, y = batch
        probabilities = model(data)
        print(probabilities)
        print(y)
        break
