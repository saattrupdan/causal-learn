'''PyTorch dataloader to load correlation matrices and their CPDAGs'''

from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import numpy as np
import scipy
import torch

from dag_sampler import DAGSampler
from gaussian_sampler import GaussianDataSampler
from config import Config


class CPDAGDataset(IterableDataset):
    '''PyTorch dataset to load correlation matrices and their CPDAGs.

    Args:
        config (Config):
            Config object containing the configuration parameters.

    Attributes:
        num_variables (int):
            Number of variables in the dataset.
        num_data_points_range (tuple):
            Range of the number of data points in the dataset.
    '''

    def __init__(self, config: Config):
        self.num_variables_range = config.num_variables_range
        self.num_data_points = config.num_data_points
        self._dag_sampler = DAGSampler(config)
        self._gaussian_sampler = GaussianDataSampler(config)

        # Set up the random number generator
        self._rng = np.random.default_rng()

    def _iteration_fn(self):
        '''Iteration function which iterates over the dataset'''
        # Infinite loop to generate an infinite supply of data
        while True:

            # Sample the number of variables, uniformly among
            # `self.num_variables_range`
            num_variables = self._rng.integers(
                low=self.num_variables_range[0],
                high=self.num_variables_range[1] + 1
            )

            # Sample the DAG and CPDAG
            dag, cpdag = self._dag_sampler.sample(num_variables)

            # Sample the data
            data_matrix = self._gaussian_sampler.sample(dag,
                                                        self.num_data_points)

            # Calculate the correlation matrix of `data_matrix`, and convert it
            # to a PyTorch tensor of shape
            # (num_data_points * num_data_points, 1)
            corr_matrix = np.corrcoef(data_matrix)
            corr_matrix = torch.from_numpy(corr_matrix)

            # Standardise the data_matrix, as otherwise the task would become
            # too easy, in that the causal dependents would have a larger
            # variance, by construction
            # data_matrix -= data_matrix.mean(axis=0)
            # data_matrix /= data_matrix.std(axis=0)

            # Set up the "edges as nodes"
            num_pairs = scipy.special.comb(num_variables, 2, exact=True)
            node_feats = torch.zeros(num_pairs, 1)

            # Set up a matrix that enumerates the "edges as nodes" in the
            # correlation matrix. This will make it easier to locate the
            # relations between the nodes. Also populate `node_feats` with the
            # correlation values.
            node_enum = torch.zeros(num_variables, num_variables)
            idx = 0
            for i in range(num_variables):
                for j in range(i):
                    node_enum[i, j] = idx
                    node_feats[idx] = corr_matrix[i, j]
                    idx += 1
            node_enum = node_enum + node_enum.t()
            node_enum[torch.eye(num_variables).bool()] = -1

            # Populate the adjacency matrix
            adj_matrix = torch.zeros(num_pairs, num_pairs)
            for i in range(num_pairs):
                row, col = (node_enum == i).nonzero()[0]
                neighbors = set(node_enum[row, :].int().tolist() +
                                node_enum[:, col].int().tolist())
                neighbors.remove(-1)
                adj_matrix[i, list(neighbors)] = 1
                adj_matrix[list(neighbors), i] = 1

            # Convert adjacency matrix to an edge list
            edge_index = adj_matrix.gt(0).nonzero().t().long()

            # Organise the input as a PyG graph, to be inputted to the model
            graph_data = Data()
            graph_data.x = node_feats
            graph_data.edge_index = edge_index

            # Convert the cpdag adjacency matrix to a PyTorch tensor
            cpdag = torch.from_numpy(cpdag.todense()).float()

            # Return the data
            yield graph_data, cpdag

    def __iter__(self):
        '''Iterate over the dataset'''
        return self._iteration_fn()


if __name__ == '__main__':
    from tqdm.auto import tqdm
    from itertools import islice
    from torch_geometric.loader import DataLoader
    from config import Config

    config = Config()
    dataset = CPDAGDataset(config)
    dataloader = DataLoader(dataset)

    for batch in dataloader:
        data, y = batch
        print(data)
        print(y)
        break

    with tqdm(total=100) as pbar:
        for batch in islice(dataloader, 100):
            pbar.update()
