'''Class that samples Gaussian data according to a DAG'''

import numpy as np
from typing import Optional, Union, Tuple, List
import multiprocessing as mp
from tqdm.auto import tqdm

from config import Config


class GaussianDataSampler:
    '''Class that samples Gaussian data according to DAGs.

    Args:
        config (Config): The configuration object.

    Attributes:
        num_data_points (int): The number of data points to sample.
    '''

    def __init__(self, config: Config):
        try:
            self.num_data_points = config.num_data_points
        except AttributeError:
            self.num_data_points = None

        # Set up the random number generator
        self._rng = np.random.default_rng()

    def _sample_noise(self, num_data_points: int) -> float:
        '''Samples a noise value.

        This samples sigma ~ Unif[0.5, 2] and epsilon ~ N(0, sigma^2), and
        returns epsilon.

        Args:
            num_data_points (int):
                The number of data points to sample.

        Returns:
            float: The sampled noise.
        '''
        sigma = self._rng.uniform(0.5, 2)
        epsilon = self._rng.normal(0, sigma**2, size=num_data_points)
        return epsilon

    def _sample_regression_coefficients(self,
            size: Optional[Union[int, Tuple[int]]] = None
            ) -> Union[float, np.ndarray]:
        '''Samples a regression coefficient.

        This samples b_sign from {-1, 1} with probability 0.4 and 0.6,
        respectively. It then samples b_val from Unif[0.1, 2] and returns the
        product of these.

        Args:
            size (int, pair of ints or None, optional):
                The size of the array to sample. If None then a single value
                will be sampled. Defaults to None.

        Returns:
            float or NumPy array:
                The sampled coefficient(s).
        '''
        # Sample the sign
        b_sign = self._rng.choice([-1, 1], p=[0.4, 0.6], size=size)

        # Sample the value
        b_val = self._rng.uniform(0.1, 2, size=size)

        # Return the product
        return b_sign * b_val

    def sample(self,
               dag: np.ndarray,
               num_data_points: Optional[int] = None) -> np.ndarray:
        '''Samples data according to the DAG.

        This samples sigma ~ Unif[0.5, 2] and epsilon ~ N(0, sigma^2) for each
        node. If the node has any parents then we sample, for each parent,
        b_val ~ Unif[0.1, 2] and b_sgn ~ {-1, 1} with probabilities 0.4 and
        0.6, respectively. With these we define beta = b_sgn * b_val and add to
        beta * X_parent, with X_parent being the data sampled for the parent.
        We repeat this for all the parents.

        Args:
            dag (NumPy array):
                The DAG to sample data from, as an adjacency matrix of shape
                (num_variables, num_variables). This is expected to be sampled
                from DAGSampler, in that it is a lower triangular matrix.
            num_data_points (int, optional):
                The number of data points to sample. If None then
                `self.num_data_points` will be used. Defaults to None.

        Returns:
            NumPy array:
                The sampled data, organised as a NumPy array with shape
                (num_nodes, num_data_points).

        Raises:
            ValueError:
                If `num_data_points` is None and `self.num_data_points` is
                also None.
        '''
        # Raise an error if both `num_data_points` and `self.num_data_points`
        # are None
        if num_data_points is None and self.num_data_points is None:
            raise ValueError('Either `num_data_points` or '
                             '`self.num_data_points` must be set.')

        # If `num_data_points` is None then use `self.num_data_points`
        if num_data_points is None:
            num_data_points = self.num_data_points

        # Initialise the data array
        data = np.zeros((dag.shape[0], num_data_points))

        # Sample the data
        for var_idx in range(dag.shape[0]):

            # Sample the data for the variable and store it
            noise = self._sample_noise(num_data_points)
            data[var_idx, :] = noise

            # Get the parents of the variable
            parents = np.nonzero(dag[var_idx, :var_idx])[0]

            # If the variable has parents then compute the parents'
            # contribution to the data
            if parents.shape[0] > 0:

                # Sample regression coefficients for the parents
                coeffs = self._sample_regression_coefficients(size=len(parents))

                # Compute the contribution from the parents
                prod = np.expand_dims(coeffs, -1) * data[parents, :]
                parents_contrib = np.sum(prod, axis=0)

                # Add the parents' contribution to the data
                data[var_idx, :] += parents_contrib

        # Return the data
        return data

    def sample_many(self,
                    dags: List[np.ndarray],
                    num_workers: int = -1) -> List[np.ndarray]:
        '''Samples data according to the DAG.

        Args:
            dags (list of NumPy arrays):
                The DAGs to sample data from, as an adjacency matrix of shape
                (num_variables, num_variables). This is expected to be sampled
                from DAGSampler, in that it is a lower triangular matrix.
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


if __name__ == '__main__':
    from dag_sampler import DAGSampler
    from config import Config
    import time

    config = Config()
    dag_sampler = DAGSampler(config)
    gaussian_sampler = GaussianDataSampler(config)

    t0 = time.time()
    dag, cpdag = dag_sampler.sample()
    print(f'Time to sample DAG: {time.time() - t0}')

    t0 = time.time()
    samples = gaussian_sampler.sample(dag)
    print(f'Time to sample data: {time.time() - t0}')

    gaussian_sampler.sample_many([dag] * 1_000)
