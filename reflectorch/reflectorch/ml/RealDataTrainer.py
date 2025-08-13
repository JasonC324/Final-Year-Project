from reflectorch.ml import Trainer 
import torch
from torch import nn

class RealDataTrainer(Trainer):
    def __init__(self, model, loader, **trainer_kwargs):
        super().__init__(model=model, loader=loader, **trainer_kwargs)
        self.loader = loader

    def get_batch_by_idx(self, batch_num):
        return self.loader.get_batch(self.batch_size)

    def get_loss_dict(self, batch_data):
        self.criterion = nn.MSELoss(reduction='none')
        
        curves = batch_data["scaled_curves"]
        targets = batch_data["scaled_params"]
        scaled_bounds = batch_data["scaled_bounds"]
        q_value = batch_data["q_values"]

        param = next(self.model.parameters())
        model_device = param.device
        model_dtype = param.dtype

        curves = curves.to(device=model_device, dtype=model_dtype)
        q_value = q_value.to(device=model_device, dtype=model_dtype)
        targets = targets.to(device=model_device, dtype=model_dtype)
        scaled_bounds = scaled_bounds.to(device=model_device, dtype=model_dtype)

        preds = self.model(curves, scaled_bounds, q_values=q_value)

        n_params = targets.shape[-1]
        b_min = scaled_bounds[..., :n_params]
        b_max = scaled_bounds[..., n_params:]
        interval_width = b_max - b_min

        base_loss = self.criterion(preds, targets)
        
        width_factors = (interval_width / 2) ** 2

        loss = (width_factors * base_loss).mean()

        return {"loss": loss}

