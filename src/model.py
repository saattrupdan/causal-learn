'''GNN model used to predict CPDAG from a correlation matrix'''

import torch_geometric.nn as tgnn
from torch_geometric.data import HeteroData
from torch_geometric.utils.dropout import dropout_adj
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Union


class CausalDiscoverer(nn.Module):
    '''GNN model used to predict CPDAG from a correlation matrix.

    Args:
        dim (int):
            The node dimension.
        dropout (float, optional):
            The dropout rate. Defaults to 0.5.

    Attributes:
        dim (int):
            The node dimension.
        dropout (float):
            The dropout rate.
        conv1 (torch.nn.Conv1d):
            The first convolutional layer.
        conv2 (torch.nn.Conv1d):
            The second convolutional layer.
    '''
    def __init__(self, dim: int, dropout: float = 0.5):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.conv1 = tgnn.HeteroConv({
            ('feat', 'correlated_with', 'feat'): tgnn.GINEConv(
                nn.Sequential(nn.Linear(1, dim),
                              nn.LayerNorm(dim),
                              nn.GELU(),
                              nn.Dropout(dropout),
                              nn.Linear(dim, dim)),
                edge_dim=1),
            ('feat', 'implies', 'feat'): tgnn.GINConv(
                nn.Sequential(nn.Linear(1, dim),
                              nn.LayerNorm(dim),
                              nn.GELU(),
                              nn.Dropout(dropout),
                              nn.Linear(dim, dim)))},
            aggr='sum')
        self.conv2 = tgnn.HeteroConv({
            ('feat', 'correlated_with', 'feat'): tgnn.GINEConv(
                nn.Sequential(nn.Linear(dim, dim),
                              nn.LayerNorm(dim),
                              nn.GELU(),
                              nn.Dropout(dropout),
                              nn.Linear(dim, dim)),
                edge_dim=1),
            ('feat', 'implies', 'feat'): tgnn.GINConv(
                nn.Sequential(nn.Linear(dim, dim),
                              nn.LayerNorm(dim),
                              nn.GELU(),
                              nn.Dropout(dropout),
                              nn.Linear(dim, dim)))},
            aggr='sum')
        self.edge_mlp = nn.Sequential(nn.Linear(dim * 2, dim),
                                      nn.LayerNorm(dim),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(dim, 1),
                                      nn.Sigmoid())

        # Initialise a random generator
        self.rng = np.random.default_rng()

    def forward(self, data: HeteroData) -> torch.Tensor:
        '''Get node features from a correlation/causal graph.

        Args:
            data (PyG HeteroData object):
                The graph data, containing 'feat' as the only node type, and
                'correlated_with' and 'implies' as the edge types.

        Returns:
            PyTorch tensor:
                The edge probabilities
        '''
        # Drop causal edges if training
        if self.training:

            # Sample causal dropout rate from Unif[0, 1], using `self.rng`
            causal_dropout = self.rng.uniform(0, 1)

            # Dropout causal edges with probability `causal_dropout`
            causal_edge_index = data['feat', 'implies', 'feat'].edge_index
            edge_index, _ = dropout_adj(edge_index=causal_edge_index,
                                        p=causal_dropout,
                                        training=self.training)
            data['feat', 'implies', 'feat'].edge_index = edge_index

        # Apply convolutional layers to get the node features
        x_dict = self.conv1(data.x_dict,
                            data.edge_index_dict,
                            data.edge_attr_dict)
        x_dict = self.conv2(x_dict,
                            data.edge_index_dict,
                            data.edge_attr_dict)

        # Extract edge features from the node features. This concatenates every
        # pair of node features in `x`, ending up with a tensor of shape
        # (num_nodes, num_nodes, `self.dim` * 2)
        x = x_dict['feat']
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
                threshold: float = 0.5) -> torch.Tensor:
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
            graph_data['feat'].x = torch.ones((num_feats, 1))
            graph_data['feat', 'correlated_with', 'feat'].edge_index = \
                torch.ones((2, num_feats * num_feats)).long()
            graph_data['feat', 'correlated_with', 'feat'].edge_attr = \
                torch.tensor(corr_matrix).view(num_feats * num_feats, 1)
            graph_data['feat', 'implies', 'feat'].edge_index = \
                torch.empty(2, 0)

            # Loop until no more causal edges are added
            while True:

                # Get the edge probabilities
                edge_probs = self.forward(graph_data)

                # If all probabilities are below the threshold then halt
                if edge_probs.max() < threshold:
                    break

                # Otherwise, add a causal edge with the highest probability
                else:
                    # Get the node pair whose edge probability is largest
                    src, tgt = edge_probs.argmax(dim=-1)

                    # Get the existing causal edge indices
                    causal_edge_index = (graph_data['feat', 'implies', 'feat']
                                         .edge_index)

                    # Add the new causal edge
                    causal_edge_index = torch.cat((causal_edge_index,
                                                   torch.tensor([[src, tgt]])),
                                                  dim=1)
                    graph_data['feat', 'implies', 'feat'].edge_index = \
                        causal_edge_index


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
