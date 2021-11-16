'''Evaluate a trained model'''

from pathlib import Path
from typing import Dict
import json
import torch
import torchmetrics as tm
from tqdm.auto import tqdm
from collections import defaultdict

from model import CausalDiscoverer
from dag_sampler import DAGSampler
from gaussian_sampler import GaussianDataSampler
from config import Config


def evaluate(model_dir: str,
             num_samples: int = 100,
             num_variables: int = 5,
             threshold: float = 0.5) -> Dict[str, float]:
    '''Evaluate a trained model.

    Args:
        model_dir (str):
            Path to the model directory.
        num_variables (int):
            Number of variables in the evaluation dataset.
        threshold (float):
            Threshold for the evaluation.

    Returns:
        dict:
            Keys are the names of the metrics, and values are the values of
            the metrics.

    Raises:
        FileNotFoundError:
            If the model directory does not exist.
    '''
    # Ensure that `model_dir` is a Path
    model_dir = Path(model_dir)

    # Raise an error if the model directory does not exist
    if not model_dir.exists():
        raise FileNotFoundError(f'Model directory {model_dir} does not exist')

    # Set up config and model paths
    config_path = model_dir / 'config.json'
    model_path = model_dir / 'model.pt'

    # Load the config
    with config_path.open('r') as f:
        config = Config(**json.load(f))

    # Set config.num_variables
    config.num_variables = num_variables

    # Initialise the metrics
    f1_metric = tm.F1(num_classes=2, average=None)
    precision_metric = tm.Precision(num_classes=2, average=None)
    recall_metric = tm.Recall(num_classes=2, average=None)
    specificity_metric = tm.Specificity(num_classes=2, average=None)

    # Initialise the model
    model = CausalDiscoverer(config)

    # Load the model weights
    model.load_state_dict(torch.load(str(model_path)))

    # Initialise the DAG sampler
    dag_sampler = DAGSampler(config)

    # Initialise the Gaussian sampler
    gaussian_sampler = GaussianDataSampler(config)

    # Sample DAGs
    dags_and_cpdags = [dag_sampler.sample()
                       for _ in tqdm(range(num_samples),
                                     desc='Sampling DAGs and CPDAGs')]
    dags = [dag_and_cpdag[0] for dag_and_cpdag in dags_and_cpdags]
    cpdags = [dag_and_cpdag[1].todense() for dag_and_cpdag in dags_and_cpdags]

    # Sample Gaussian data
    data_matrices = [gaussian_sampler.sample(dag=dag)
                     for dag in tqdm(dags, desc='Sampling data')]

    # Compute predicted adjacency matrices for the data matrices
    adj_matrices = list()
    for data_matrix in tqdm(data_matrices, desc='Computing'):
        adj_matrix  = model.predict(data_matrix, threshold=threshold)
        adj_matrices.append(adj_matrix)

    # Flatten the cpdags and the adj_matrices
    cpdags = [cpdag.flatten() for cpdag in cpdags]
    adj_matrices = [adj_matrix.flatten() for adj_matrix in adj_matrices]

    # Set up the cpdags and adj_matrices as tensors
    cpdags = torch.cat([torch.from_numpy(cpdag).squeeze(0).long()
                        for cpdag in cpdags])
    adj_matrices = torch.cat([torch.from_numpy(adj_matrix).long()
                              for adj_matrix in adj_matrices])

    # Compute the metrics
    f1 = f1_metric(cpdags, adj_matrices)[0]
    precision = precision_metric(cpdags, adj_matrices)[0]
    recall = recall_metric(cpdags, adj_matrices)[0]
    specificity = specificity_metric(cpdags, adj_matrices)[0]

    # Define a dictionary to store the metrics
    metrics = dict(f1=f1,
                   precision=precision,
                   recall=recall,
                   specificity=specificity)

    # Return the metrics
    return metrics


if __name__ == '__main__':
    model_dir = 'models/model1'
    metrics = evaluate(model_dir,
                       num_variables=5,
                       num_samples=100,
                       threshold=0.3)
    print(metrics)
