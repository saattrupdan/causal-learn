'''Config class used to store all hyperparameters'''

from typing import Tuple
import datetime as dt
from pydantic import BaseModel


class Config(BaseModel):
    '''Config class used to store all hyperparameters'''

    # Dataset hyperparameters
    num_variables_range: Tuple[int, int] = (10, 10)
    num_data_points: int = 10_000
    sparsity_interval: Tuple[float, float] = (0.0, 0.8)

    # Model hyperparameters
    dim: int = 256
    dropout: float = 0.0
    num_layers: int = 2
    num_heads: int = 1

    # Training hyperparameters
    num_iterations: int = 100_000
    lr: float = 3e-4
    lr_decay_factor: float = 0.9999
    batch_size: int = 32
    threshold: float = 0.4

    # Metric hyperparameters
    ema_decay: float = 0.999  # Corresponding to averaging last 1000 steps

    # IO hyperparameters
    save_model: bool = False
    _datetime = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_dir: str = f'./models/{_datetime}-model'
