'''PyTorch dataloader to load correlation matrices and their CPDAGs'''

from torch.utils.data import IterableDataset
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import numpy as np
import torch
from typing import Optional, Tuple
from dag_sampler import DAGSampler
from gaussian_sampler import GaussianDataSampler


class CorrDataset(IterableDataset):
    '''PyTorch dataset to load correlation matrices and their CPDAGs'''

    def __init__(self,
                 num_variables_range: Tuple[int, int] = (3, 5),
                 num_data_points_range: Tuple[int, int] = (100, 1000),
                 random_seed: Optional[int] = None):
        self.num_variables_range = num_variables_range
        self.num_data_points_range = num_data_points_range
        self._dag_sampler = DAGSampler(random_seed=random_seed)
        self._gaussian_sampler = GaussianDataSampler(random_seed=random_seed)

        # Set up the random number generator
        self.rng = np.random.default_rng(seed=random_seed)

    def _iteration_fn(self):
        '''Iteration function which iterates over the dataset'''
        # Infinite loop to generate an infinite supply of data
        while True:

            # Sample the number of variables, uniformly among
            # `self.num_variables_range`
            num_variables = self.rng.integers(
                low=self.num_variables_range[0],
                high=self.num_variables_range[1]
            )

            # Sample the number of data points, uniformly among
            # `self.num_data_points_range`
            num_data_points = self.rng.integers(
                low=self.num_data_points_range[0],
                high=self.num_data_points_range[1]
            )

            # Sample the DAG and CPDAG
            dag, cpdag = self._dag_sampler.sample(num_variables)

            # Sample the data
            data_matrix = self._gaussian_sampler.sample(dag, num_data_points)

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
