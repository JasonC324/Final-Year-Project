from reflectorch.data_generation.dataset import BasicDataset 
import torch
from torch import Tensor
import itertools
import random

from typing import Dict, Union
from reflectorch.data_generation.priors import BasicParams
from reflectorch.ml.basic_trainer import TrainerCallback
from reflectorch.data_generation.priors import SubpriorParametricSampler

BATCH_DATA_TYPE = Dict[str, Union[Tensor, BasicParams]]

class InputPreMadeDataset(BasicDataset,TrainerCallback):
    def __init__(
        self, q_values, curves, scaled_curves, params, scaled_params, scaled_bounds, device=None, dtype=None):
        super().__init__(
            q_generator = None,
            prior_sampler = None,
            intensity_noise = None,
            q_noise = None,
            curves_scaler = None,
            calc_denoised_curves = False,
            calc_nonsmeared_curves = False,
            smearing = None
        )

        if not isinstance(q_values, torch.Tensor):
            q_values = torch.tensor(q_values, dtype=dtype)
        if not isinstance(scaled_curves, torch.Tensor):
            scaled_curves = torch.tensor(scaled_curves, dtype=dtype)
        if not isinstance(scaled_params, torch.Tensor):
            scaled_params = torch.tensor(scaled_params, dtype=dtype)
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=dtype)
        
        if not isinstance(curves, torch.Tensor):
            self.curves = torch.tensor(curves, dtype=dtype)
        if self.curves.is_complex():
            self.curves = self.curves.real
            self.curves = self.curves.float()
        if not isinstance(self.curves, torch.Tensor):
            self.curves = torch.tensor(self.curves, dtype=dtype)
        
        self.scaled_curves = scaled_curves.to(device=device, dtype=dtype)
        if self.scaled_curves.is_complex():
            self.scaled_curves = self.scaled_curves.real
            self.scaled_curves = self.scaled_curves.float()


        self.q_values = q_values.to(device=device, dtype=dtype)
        self.scaled_params = scaled_params.to(device=device, dtype=dtype)

        self.params = params.to(device=device, dtype=dtype)
        if self.params.is_complex():
            self.params = self.params.real
            self.params = self.params.float()


        self.scaled_bounds = scaled_bounds.to(device=device, dtype=dtype)
        self._iter = self._shuffled_cycle()

        self.pos = 0
        self.n = self.scaled_bounds.shape[0]

    def start_training(self, trainer):
        self._iter = self._shuffled_cycle()

    def end_batch(self, trainer, batch_num):
        return False

    def get_batch(self, batch_size: int)-> BATCH_DATA_TYPE:
        index = [next(self._iter) for _ in range(batch_size)]
        batch_data = {}

        batch_data['params'] = self.params[index]
        batch_data['scaled_params'] = self.scaled_params[index]
        batch_data['q_values'] = self.q_values[index]
        batch_data['scaled_curves'] = self.scaled_curves[index]
        batch_data['scaled_bounds'] = self.scaled_bounds[index]
        batch_data['curves'] = self.curves[index]

        self.pos = self.pos + batch_size
        if self.pos >= self.n:
            self._iter = self._shuffled_cycle()
            self.pos = 0

        return batch_data
    
    def __len__(self):
        return len(self.scaled_curves)
    
    def _shuffled_cycle(self):
        n = self.params.shape[0]
        while True:
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                yield idx
