'''Config class used to store all hyperparameters'''

from typing import Tuple
import datetime as dt


class Config:
    '''Config class used to store all hyperparameters'''

    # Dataset hyperparameters
    num_variables_range: Tuple[int, int] = (2, 10)
    num_data_points : int = 10_000

    # Model hyperparameters
    dim: int = 250
    dropout: float = 0.0

    # Training hyperparameters
    num_iterations: int = 1_000_000
    lr: float = 3e-4
    batch_size: int = 1
    threshold: float = 0.5

    # Metric hyperparameters
    ema_decay: float = 0.99

    # Miscellaneous hyperparameters
    _datetime = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_dir: str = f'./models/{_datetime}-model'

    def __init__(self, **kwargs):
        # Update the config with the values of the class variables
        class_vars = {key: getattr(self, key) for key in dir(self)
                      if not key.startswith('__')}
        self.__dict__.update(class_vars)

        # Update the config with the values of the kwargs
        self.__dict__.update(kwargs)


    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __getitem__(self, key: str):
        '''Get a hyperparameter.

        Args:
            key (str): The key of the hyperparameter.

        Returns:
            The hyperparameter.
        '''
        return self.__dict__[key]
