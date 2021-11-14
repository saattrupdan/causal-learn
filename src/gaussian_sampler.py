'''Class that samples Gaussian data according to a DAG'''

import numpy as np
from causaldag import DAG
from typing import Optional, List
import multiprocessing as mp
from tqdm.auto import tqdm


class GaussianDataSampler:
    '''Class that samples Gaussian data according to DAGs.

    Args:
        num_data_points (int):
            The number of data points to sample for each node in the DAGs.
        random_seed (int or None, optional):
            A random seed to be used for setting up the sampling. If set then
            the samples will still be different, but the results will be
            reproducible. If set to None then no random seed will be set.
            Defaults to None.

    Attributes:
        random_seed (int or None): Random seed used for sampling.
        rng (NumPy Generator): Random number generator.
    '''

    def __init__(self,
                 num_data_points: int,
                 random_seed: Optional[int] = None):
        self.num_data_points = num_data_points
        self.random_seed = random_seed

        # Set up the random number generator
        self.rng = np.random.default_rng(seed=random_seed)

    def sample(self, dag: DAG) -> np.ndarray:
        '''Samples data according to the DAG.

        Args:
            dag (DAG): The DAG to sample data from.

        Returns:
            NumPy array:
                The sampled data, organised as a NumPy array with shape
                (num_nodes, self.num_data_points).
        '''
        raise NotImplementedError

    def sample_many(self,
                    dags: List[DAG],
                    num_workers: int = -1) -> List[np.ndarray]:
        '''Samples data according to the DAG.

        Args:
            dags (list of DAG):
                The DAGs to sample data from.
            num_workers (int, optional):
                Number of workers to use for parallel sampling. If set to -1
                then the number of workers will be set to the number of CPUs
                available on the machine. Defaults to -1.

        Returns:
            list of NumPy arrays:
                The sampled data, organised as a list of NumPy arrays, each of
                shape (num_nodes, self.num_data_points).
        '''
        # Set the number of workers to the number of CPUs available on the
        # machine if the number of workers is -1
        if num_workers == -1:
            num_workers = mp.cpu_count()

        # Set up multiprocessing if `num_workers` is not 1
        if num_workers == 1:
            return [self.sample(dag) for dag in dags]

        # Otherwise compute samples in parallel and return the results
        else:
            with mp.Pool(processes=num_workers) as pool:
                imap = pool.imap_unordered(self.sample, dags)
                with tqdm(imap, total=len(dags)) as pbar:
                    samples = [sample for sample in pbar]
            return samples
