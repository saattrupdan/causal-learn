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

    def _sample_noise(self) -> float:
        '''Samples a noise value.

        This samples sigma ~ Unif[0.5, 2] and epsilon ~ N(0, sigma^2), and
        returns epsilon.

        Returns:
            float: The sampled noise.
        '''
        sigma = self.rng.uniform(0.5, 2)
        epsilon = self.rng.normal(0, sigma**2)
        return epsilon

    def _sample_regression_coefficient(self) -> float:
        '''Samples a regression coefficient.

        This samples b_sign from {-1, 1} with probability 0.4 and 0.6,
        respectively. It then samples b_val from Unif[0.1, 2] and returns the
        product of these.

        Returns:
            float: The sampled coefficient.
        '''
        # Sample the sign
        b_sign = self.rng.choice([-1, 1], p=[0.4, 0.6])

        # Sample the value
        b_val = self.rng.uniform(0.1, 2)

        # Return the product
        return b_sign * b_val

    def _sample_data(self,
                     node: int,
                     dag: DAG,
                     data: np.ndarray) -> np.ndarray:
        '''Samples data for a node.

        Args:
            node (int): The node to sample data for.
            dag (DAG): The DAG to sample data from.
            data (NumPy array): The sampled data.

        Returns:
            NumPy array: The sampled data.
        '''
        # Get the parents of the node
        parents = dag.parents_of(node)

        # Get the noise value for the node
        node_data = np.array([self._sample_noise()
                              for _ in range(self.num_data_points)])

        # Add the contributions from the parents
        for parent in parents:

            # Sample the regression coefficients
            coeff = self._sample_regression_coefficient()

            # Get the data from the parent
            parent_data = self._sample_data(parent, dag, data)

            # Compute the contribution from the parent
            parent_contrib = coeff * parent_data[parent]

            # Add the contribution from the parent to the node data
            node_data += parent_contrib

        # Add the node data to the data array
        data[node] = node_data

        # Return the data
        return data

    def sample(self, dag: DAG) -> np.ndarray:
        '''Samples data according to the DAG.

        This samples sigma ~ Unif[0.5, 2] and epsilon ~ N(0, sigma^2) for each
        node. If the node has any parents then we sample, for each parent,
        b_val ~ Unif[0.1, 2] and b_sgn ~ {-1, 1} with probabilities 0.4 and
        0.6, respectively. With these we define beta = b_sgn * b_val and add to
        beta * X_parent, with X_parent being the data sampled for the parent.
        We repeat this for all the parents.

        Args:
            dag (DAG): The DAG to sample data from.

        Returns:
            NumPy array:
                The sampled data, organised as a NumPy array with shape
                (num_nodes, self.num_data_points).
        '''
        # Collect the sink nodes in the DAG
        sink_nodes = list(dag.sinks())

        # Add a new node to the DAG, which becomes the unique sink node
        new_node_index = dag.nnodes
        dag.add_node(new_node_index)
        for sink_node in sink_nodes:
            dag.add_arc(sink_node, new_node_index)

        # Initialise the data array
        data = np.zeros((dag.nnodes, self.num_data_points))

        # Sample the data from the new unique sink node, which will recursively
        # sample the data from all the parents
        data = self._sample_data(node=new_node_index, dag=dag, data=data)

        # Remove the new node from the data
        data = data[:-1]

        # Return the data
        return data

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


if __name__ == '__main__':
    from dag_sampler import DAGSampler

    dag_sampler = DAGSampler(num_variables=5, random_seed=4242)
    gaussian_sampler = GaussianDataSampler(num_data_points=10, random_seed=0)

    dag, cpdag = dag_sampler.sample()
    samples = gaussian_sampler.sample(dag)

    print(samples.shape)

    gaussian_sampler.sample_many([dag] * 100_000)
