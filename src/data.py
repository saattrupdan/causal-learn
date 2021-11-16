'''PyTorch dataloader to load correlation matrices and their CPDAGs'''

from torch.utils.data import IterableDataset
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import numpy as np
import torch

from dag_sampler import DAGSampler
from gaussian_sampler import GaussianDataSampler
from config import Config


class CorrDataset(IterableDataset):
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
                high=self.num_variables_range[1]
            )

            # Sample the DAG and CPDAG
            dag, cpdag = self._dag_sampler.sample(num_variables)

            # Sample the data
            data_matrix = self._gaussian_sampler.sample(dag,
                                                        self.num_data_points)

            # Compute the correlation matrix of `data_matrix`
            corr_matrix = np.corrcoef(data_matrix.T)

            # Store the number of features
            num_feats = data_matrix.shape[1]

            # Get the edge_index of the CPDAG
            cpdag_edge_idx, _ = from_scipy_sparse_matrix(cpdag)

            # Organise the input as a PyG graph, to be inputted to the model
            graph_data = HeteroData()
            graph_data['feat'].x = torch.ones((num_feats, 1))
            graph_data['feat', 'correlated_with', 'feat'].edge_index = \
                torch.ones((2, num_feats * num_feats)).long()
            graph_data['feat', 'correlated_with', 'feat'].edge_attr = \
                (torch.tensor(corr_matrix)
                      .view(num_feats * num_feats, 1)
                      .float())
            graph_data['feat', 'implies', 'feat'].edge_index = cpdag_edge_idx

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

    dataset = CorrDataset()
    dataloader = DataLoader(dataset)

    for batch in dataloader:
        data = batch
        print(data)
        break

    with tqdm(total=100) as pbar:
        for batch in islice(dataloader, 100):
            pbar.update()
