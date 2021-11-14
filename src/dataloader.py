'''PyTorch dataloader to load correlation matrices and their CPDAGs'''

from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import torch
from typing import Optional, Tuple, Iterable
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

            # Return the data
            yield torch.from_numpy(corr_matrix), torch.from_numpy(cpdag)

    def __iter__(self):
        '''Iterate over the dataset'''
        return self._iteration_fn()


if __name__ == '__main__':
    from tqdm.auto import tqdm
    from itertools import islice
    dataset = CorrDataset()
    dataloader = DataLoader(dataset)
    with tqdm(total=100) as pbar:
        for batch in islice(dataloader, 100):
            pbar.update()
