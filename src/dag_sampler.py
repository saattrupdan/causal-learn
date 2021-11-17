'''Class that samples a Directed Acyclic Graph (DAG)'''

import numpy as np
from scipy import sparse
from causaldag import DAG
from typing import Optional, Tuple, List
import multiprocessing as mp
from tqdm.auto import tqdm

from config import Config


class DAGSampler:
    '''Class that samples Directed Acyclic Graphs (DAGs).

    Args:
        config (Config):
            Configuration object.

    Attributes:
        num_variables (int or None): Number of variables in the DAGs.
    '''
    def __init__(self, config: Config):
        try:
            self.num_variables = config.num_variables
        except AttributeError:
            start = config.num_variables_range[0]
            end = config.num_variables_range[1]
            if start + 1 == end
                self.num_variables = start
            else:
                self.num_variables = None

        # Set up the random number generator
        self._rng = np.random.default_rng()


    def sample(self,
               num_variables: Optional[int] = None) -> Tuple[DAG, np.ndarray]:
        '''Sample a DAG.

        Args:
            num_variables (int or None, optional):
                Number of variables in the DAG. If set to None, then
                self.num_variables will be used. Defaults to None.

        Returns:
            tuple of DAG and SciPy sparse matrix:
                A tuple of a sampled DAG and its associated CPDAG, the latter
                organised as a SciPy sparse matrix of its adjacency matrix.

        Raises:
            ValueError:
                If both `num_variables` and `self.num_variables` are None.
        '''
        # Raise an error if both `num_variables` and `self.num_variables` are
        # None
        if num_variables is None and self.num_variables is None:
            raise ValueError('Either `num_variables` or `self.num_variables` '
                             'must be set.')

        # If `num_variables` is None then use `self.num_variables`
        if num_variables is None:
            num_variables = self.num_variables

        # Initialize the adjacency matrix as a lower triangular matrix of ones
        adj_matrix = np.tril(np.ones((num_variables, num_variables)))

        # Sample the sparsity from Unif[0, 0.8]
        sparsity = self._rng.uniform(0, 0.8)

        # For each non-zero entry in the adjacency matrix, sample a Bernoulli
        # variable with probability `sparsity` and set the entry to 0 if the
        # Bernoulli variable is 1. This is equivalent to setting the entry to
        # 0 with probability `1 - sparsity`, which is what we are doing here.
        non_zero_shape = adj_matrix[adj_matrix != 0].shape
        adj_matrix[adj_matrix != 0] = self._rng.binomial(n=1,
                                                         p=1-sparsity,
                                                         size=non_zero_shape)

        # Convert the adjacency matrix to a DAG
        dag = DAG.from_amat(adj_matrix)

        # Extract the CPDAG from the DAG, as a sparse adjacency matrix
        cpdag = sparse.csr_matrix(dag.cpdag().to_amat()[0])

        return dag, cpdag

    def sample_many(self,
                    num_samples: int,
                    num_workers: int = -1) -> List[Tuple[DAG, np.ndarray]]:
        '''Sample many DAGs.

        Args:
            num_samples (int):
                Number of DAGs to sample.
            num_workers (int, optional):
                Number of workers to use for parallel sampling. If set to -1
                then the number of workers will be set to the number of CPUs
                available on the machine. Defaults to -1.

        Returns:
            list of tuples of DAG and NumPy array:
                A list of tuples of sampled DAGs and their associated CPDAGs,
                the latter organised as a NumPy array of their adjacency
                matrices.
        '''
        # Set the number of workers to the number of CPUs available on the
        # machine if the number of workers is -1
        if num_workers == -1:
            num_workers = mp.cpu_count()

        # Set up multiprocessing if `num_workers` is not 1
        if num_workers == 1:
            return [self.sample() for _ in range(num_samples)]

        # Otherwise compute samples in parallel and return the results
        else:
            with mp.Pool(processes=num_workers) as pool:
                imap = pool.imap_unordered(self.sample, range(num_samples))
                with tqdm(imap, total=num_samples) as pbar:
                    samples = [sample for sample in pbar]
            return samples


if __name__ == '__main__':
    from config import Config

    config = Config()
    dag_sampler = DAGSampler(config)

    dag, cpdag = dag_sampler.sample()
    dag_sampler.sample_many(1_000)
