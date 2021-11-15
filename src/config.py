'''Config class used to store all hyperparameters'''


class Config:
    '''Config class used to store all hyperparameters'''

    # Set default hyperparameters
    dim: int = 100
    num_iterations: int = 1_000_000
    dropout: float = 0.0
    lr: float = 3e-4
    ema_decay: float = 0.999
    batch_size: int = 32
    model_path: str = './models/model.pt'
    threshold: float = 0.5

    def __init__(self, **kwargs):
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
