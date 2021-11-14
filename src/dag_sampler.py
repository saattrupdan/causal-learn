'''Class that samples a Directed Acyclic Graph (DAG)'''

import numpy as np
from causaldag import DAG
from typing import Optional, Tuple, List
import multiprocessing as mp
from tqdm.auto import tqdm


class DAGSampler:
    '''Class that samples Directed Acyclic Graphs (DAGs).

    Args:
        num_variables (int):
            Number of variables in the DAGs.
        random_seed (int or None, optional):
            A random seed to be used for setting up the sampling. If set then
            the samples will still be different, but the results will be
            reproducible. If set to None then no random seed will be set.
            Defaults to None.

    Attributes:
        num_variables (int): Number of variables in the DAGs.
        random_seed (int or None): Random seed used for sampling.
        rng (NumPy Generator): Random number generator.
    '''
    def __init__(self, num_variables: int, random_seed: Optional[int] = None):
        self.num_variables = num_variables
        self.random_seed = random_seed

        # Set up the random number generator
        self.rng = np.random.default_rng(seed=random_seed)


    def sample(self, *_) -> Tuple[DAG, np.ndarray]:
        '''Sample a DAG.

        Returns:
            tuple of DAG and NumPy array:
                A tuple of a sampled DAG and its associated CPDAG, the latter
                organised as a NumPy array of its adjacency matrix.
        '''
        # Initialize the adjacency matrix as a lower triangular matrix of ones
        adj_matrix = np.tril(np.ones((self.num_variables, self.num_variables)))

        # Sample the sparsity from Unif[0, 0.8]
        sparsity = self.rng.uniform(0, 0.8)

        # For each non-zero entry in the adjacency matrix, sample a Bernoulli
        # variable with probability `sparsity` and set the entry to 0 if the
        # Bernoulli variable is 1. This is equivalent to setting the entry to
        # 0 with probability `1 - sparsity`, which is what we are doing here.
        non_zero_shape = adj_matrix[adj_matrix != 0].shape
        adj_matrix[adj_matrix != 0] = self.rng.binomial(n=1,
                                                        p=1-sparsity,
                                                        size=non_zero_shape)


        # Convert the adjacency matrix to a DAG
        dag = DAG.from_amat(adj_matrix)

        # Extract the CPDAG from the DAG, as an adjacency matrix
        cpdag = dag.cpdag().to_amat()[0]

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
    # Test the DAG sampler
    dag_sampler = DAGSampler(num_variables=5, random_seed=4242)
    print(dag_sampler.sample())
    dag_sampler.sample_many(50_000)
