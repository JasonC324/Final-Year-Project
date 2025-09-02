from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "MULTILAYER_MODELS",
    "MultilayerModel",
]


class MultilayerModel(object):
    NAME: str = ''
    PARAMETER_NAMES: Tuple[str, ...]

    def __init__(self, max_num_layers: int):
        self.max_num_layers = max_num_layers

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        raise NotImplementedError

    def from_standard_params(self, params: dict) -> Tensor:
        raise NotImplementedError


class BasicMultilayerModel1(MultilayerModel):
    NAME = 'repeating_multilayer_v1'

    PARAMETER_NAMES = (
        "d_full_rel",
        "rel_sigmas",
        "d_block",
        "s_block_rel",
        "r_block",
        "dr",
        "d3_rel",
        "s3_rel",
        "r3",
        "d_sio2",
        "s_sio2",
        "s_si",
        "r_sio2",
        "r_si",
    )

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        return multilayer_model1(parametrized_model, self.max_num_layers)


class BasicMultilayerModel2(MultilayerModel):
    NAME = 'repeating_multilayer_v2'

    PARAMETER_NAMES = (
        "d_full_rel",
        "rel_sigmas",
        "dr_sigmoid_rel_pos",
        "dr_sigmoid_rel_width",
        "d_block",
        "s_block_rel",
        "r_block",
        "dr",
        "d3_rel",
        "s3_rel",
        "r3",
        "d_sio2",
        "s_sio2",
        "s_si",
        "r_sio2",
        "r_si",
    )

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        return multilayer_model2(parametrized_model, self.max_num_layers)


class BasicMultilayerModel3(MultilayerModel):
    NAME = 'repeating_multilayer_v3'

    PARAMETER_NAMES = (
        "d_full_rel",
        "rel_sigmas",
        "dr_sigmoid_rel_pos",
        "dr_sigmoid_rel_width",
        "d_block1_rel",
        "d_block",
        "s_block_rel",
        "r_block",
        "dr",
        "d3_rel",
        "s3_rel",
        "r3",
        "d_sio2",
        "s_sio2",
        "s_si",
        "r_sio2",
        "r_si",
    )

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        return multilayer_model3(parametrized_model, self.max_num_layers)
    
class CustomMultilayerModel(MultilayerModel):
    NAME = 'repeating_multilayer_abc'

    PARAMETER_NAMES = (
        "d_sig",            # Relative total film thickness (used to center sigmoid envelope)
        "width_sigmas",             # Width of sigmoid envelope
        "k",                      # steepness of second sigmoid
        "d_A", "d_B", "d_C",      # Absolute thicknesses of A, B, C
        "s_A", "s_B", "s_C",      # Roughnesses of A, B, C
        "sld_A", "sld_B", "sld_C",      # SLDs of A, B, C
        "d_sio2",                 # SiO₂ cap thickness
        "s_sio2",                 # SiO₂ roughness
        "s_si",                   # Si substrate roughness
        "r_sio2",                 # SLD of SiO₂
        "r_si",                   # SLD of Si
    )

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        return multilayer_model_abc(parametrized_model, self.max_num_layers)


MULTILAYER_MODELS = {
    'repeating_multilayer_v1': BasicMultilayerModel1,
    'repeating_multilayer_v2': BasicMultilayerModel2,
    'repeating_multilayer_v3': BasicMultilayerModel3,
    'repeating_multilayer_abc': CustomMultilayerModel,
}


def multilayer_model1(parametrized_model: Tensor, d_full_rel_max: int = 50) -> dict:
    n = d_full_rel_max

    (
        d_full_rel,
        rel_sigmas,
        d_block,
        s_block_rel,
        r_block,
        dr,
        d3_rel,
        s3_rel,
        r3,
        d_sio2,
        s_sio2,
        s_si,
        r_sio2,
        r_si,
    ) = parametrized_model.T

    batch_size = parametrized_model.shape[0]

    r_positions = 2 * n - torch.arange(2 * n, dtype=dr.dtype, device=dr.device)[None].repeat(batch_size, 1)

    r_modulations = torch.sigmoid(-(r_positions - 2 * d_full_rel[..., None]) / rel_sigmas[..., None])

    r_block = r_block[:, None].repeat(1, n)
    dr = dr[:, None].repeat(1, n)

    sld_blocks = torch.stack([r_block, r_block + dr], -1).flatten(1)

    sld_blocks = r_modulations * sld_blocks

    d3 = d3_rel * d_block

    thicknesses = torch.cat(
        [(d_block / 2)[:, None].repeat(1, n * 2), d3[:, None], d_sio2[:, None]], -1
    )

    s_block = s_block_rel * d_block

    roughnesses = torch.cat(
        [s_block[:, None].repeat(1, n * 2), (s3_rel * d3)[:, None], s_sio2[:, None], s_si[:, None]], -1
    )

    slds = torch.cat(
        [sld_blocks, r3[:, None], r_sio2[:, None], r_si[:, None]], -1
    )

    params = dict(
        thicknesses=thicknesses,
        roughnesses=roughnesses,
        slds=slds
    )
    return params


def multilayer_model2(parametrized_model: Tensor, d_full_rel_max: int = 50) -> dict:
    n = d_full_rel_max

    (
        d_full_rel,
        rel_sigmas,
        dr_sigmoid_rel_pos,
        dr_sigmoid_rel_width,
        d_block,
        s_block_rel,
        r_block,
        dr,
        d3_rel,
        s3_rel,
        r3,
        d_sio2,
        s_sio2,
        s_si,
        r_sio2,
        r_si,
    ) = parametrized_model.T

    batch_size = parametrized_model.shape[0]

    r_positions = 2 * n - torch.arange(2 * n, dtype=dr.dtype, device=dr.device)[None].repeat(batch_size, 1)

    r_modulations = torch.sigmoid(-(r_positions - 2 * d_full_rel[..., None]) / rel_sigmas[..., None])

    r_block = r_block[:, None].repeat(1, n)
    dr = dr[:, None].repeat(1, n)

    dr_positions = r_positions[:, ::2]

    dr_modulations = torch.sigmoid(
        -(dr_positions - (2 * d_full_rel * dr_sigmoid_rel_pos)[..., None]) / dr_sigmoid_rel_width[..., None]
    )

    dr = dr * dr_modulations

    sld_blocks = torch.stack([r_block, r_block + dr], -1).flatten(1)

    sld_blocks = r_modulations * sld_blocks

    d3 = d3_rel * d_block

    thicknesses = torch.cat(
        [(d_block / 2)[:, None].repeat(1, n * 2), d3[:, None], d_sio2[:, None]], -1
    )

    s_block = s_block_rel * d_block

    roughnesses = torch.cat(
        [s_block[:, None].repeat(1, n * 2), (s3_rel * d3)[:, None], s_sio2[:, None], s_si[:, None]], -1
    )

    slds = torch.cat(
        [sld_blocks, r3[:, None], r_sio2[:, None], r_si[:, None]], -1
    )

    params = dict(
        thicknesses=thicknesses,
        roughnesses=roughnesses,
        slds=slds
    )
    return params


def multilayer_model3(parametrized_model: Tensor, d_full_rel_max: int = 30):
    n = d_full_rel_max

    (
        d_full_rel,
        rel_sigmas,
        dr_sigmoid_rel_pos,
        dr_sigmoid_rel_width,
        d_block1_rel,
        d_block,
        s_block_rel,
        r_block,
        dr,
        d3_rel,
        s3_rel,
        r3,
        d_sio2,
        s_sio2,
        s_si,
        r_sio2,
        r_si,
    ) = parametrized_model.T

    batch_size = parametrized_model.shape[0]

    r_positions = 2 * n - torch.arange(2 * n, dtype=dr.dtype, device=dr.device)[None].repeat(batch_size, 1)

    r_modulations = torch.sigmoid(
        -(
                r_positions - 2 * d_full_rel[..., None]
        ) / rel_sigmas[..., None]
    )

    dr_positions = r_positions[:, ::2]

    dr_modulations = dr[..., None] * (1 - torch.sigmoid(
        -(
                dr_positions - 2 * d_full_rel[..., None] + 2 * dr_sigmoid_rel_pos[..., None]
        ) / dr_sigmoid_rel_width[..., None]
    ))

    r_block = r_block[..., None].repeat(1, n)
    dr = dr[..., None].repeat(1, n)

    sld_blocks = torch.stack(
        [
            r_block + dr_modulations * (1 - d_block1_rel[..., None]),
            r_block + dr - dr_modulations * d_block1_rel[..., None]
        ], -1).flatten(1)

    sld_blocks = r_modulations * sld_blocks

    d3 = d3_rel * d_block

    d1, d2 = d_block * d_block1_rel, d_block * (1 - d_block1_rel)

    thickness_blocks = torch.stack([d1[:, None].repeat(1, n), d2[:, None].repeat(1, n)], -1).flatten(1)

    thicknesses = torch.cat(
        [thickness_blocks, d3[:, None], d_sio2[:, None]], -1
    )

    s_block = s_block_rel * d_block

    roughnesses = torch.cat(
        [s_block[:, None].repeat(1, n * 2), (s3_rel * d3)[:, None], s_sio2[:, None], s_si[:, None]], -1
    )

    slds = torch.cat(
        [sld_blocks, r3[:, None], r_sio2[:, None], r_si[:, None]], -1
    )

    params = dict(
        thicknesses=thicknesses,
        roughnesses=roughnesses,
        slds=slds
    )
    return params

def multilayer_model_abc(parametrized_model: Tensor, d_full_rel_max: int = 30):

    if d_full_rel_max % 3 != 0:
        raise ValueError(f"Number {d_full_rel_max} is not divisible by 3.")

    n = d_full_rel_max

    (
        d_sig,              # Relative total film thickness (used to center sigmoid envelope)
        width_sigmas,              # Width of sigmoid envelope
        k,                       # steepness of second sigmoid
        d_A, d_B, d_C,           # Absolute thicknesses of A, B, C
        s_A, s_B, s_C,           # Roughnesses of A, B, C
        sld_A, sld_B, sld_C,     # SLDs of A, B, C
        d_sio2,                  # SiO₂ cap thickness
        s_sio2,                  # SiO₂ roughness
        s_si,                    # Si substrate roughness
        r_sio2,                  # SLD of SiO₂
        r_si,                    # SLD of Si
    ) = parametrized_model.T

    batch_size = parametrized_model.shape[0]
    total_layers = n
    total_repeats = n/3

    r_positions = total_layers - torch.arange(total_layers, dtype=parametrized_model.dtype, device=parametrized_model.device)[None].repeat(batch_size, 1)

    r_modulations = torch.sigmoid(
        -(r_positions - d_sig[..., None]) / width_sigmas[..., None]
    ) 

    sld_A_rep = sld_A[:, None].repeat(1, int(total_repeats))
    sld_B_rep = sld_B[:, None].repeat(1, int(total_repeats))
    sld_C_rep = sld_C[:, None].repeat(1, int(total_repeats))

    slds_init = torch.stack([sld_A_rep, sld_B_rep, sld_C_rep], axis=2).reshape(batch_size, -1)
    center_sld = slds_init.mean()

    k = k[..., None]
    squeeze = 1 - torch.sigmoid(k * (r_positions - d_sig[..., None]))
    slds_squeezed = center_sld + squeeze * (slds_init - center_sld)

    slds_final = slds_squeezed * r_modulations

    d_A_stack = d_A[:, None].repeat(1, int(total_repeats))
    d_B_stack = d_B[:, None].repeat(1, int(total_repeats))
    d_C_stack = d_C[:, None].repeat(1, int(total_repeats))
    thickness_abc = torch.stack([d_A_stack, d_B_stack, d_C_stack], dim=-1).flatten(1)

    s_A_stack = s_A[:, None].repeat(1, int(total_repeats))
    s_B_stack = s_B[:, None].repeat(1, int(total_repeats))
    s_C_stack = s_C[:, None].repeat(1, int(total_repeats))
    roughness_abc = torch.stack([s_A_stack, s_B_stack, s_C_stack], dim=-1).flatten(1)

    thicknesses = torch.cat([thickness_abc, d_sio2[:, None]], dim=-1)
    roughnesses = torch.cat([roughness_abc, s_sio2[:, None], s_si[:, None]], dim=-1)
    slds = torch.cat([slds_final, r_sio2[:, None], r_si[:, None]], dim=-1)

    params = dict(
        thickness=thicknesses,
        roughness=roughnesses,
        sld=slds
    )

    return params
