import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def scale_params(thickness_array,roughness_array,slds_array,background_array):
    roughness_min = 0.0
    roughness_max = 60.0

    sld_min = slds_array.min()
    sld_max = slds_array.max()

    thickness_min, thickness_max = 0.0, 500.0

    background_min, background_max = 1e-7, 1e-5

    print(roughness_min)
    print(roughness_max)
    print(sld_min)
    print(sld_max)


    thicknesses_scaled = (thickness_array - thickness_min) / (thickness_max - thickness_min)
    thicknesses_scaled = 2.0 * (thicknesses_scaled - 0.5)
    roughnesses_scaled = (roughness_array - roughness_min) / (roughness_max - roughness_min)
    roughnesses_scaled = 2.0 * (roughnesses_scaled - 0.5)
    slds_scaled = (slds_array - sld_min) / (sld_max - sld_min)
    slds_scaled = 2.0 * (slds_scaled - 0.5)

    background_scaled = (background_array - background_min) / (background_max - background_min)
    background_scaled = 2.0 * (background_scaled - 0.5)

    thicknesses_scaled = np.clip(thicknesses_scaled, -1, 1)
    roughnesses_scaled = np.clip(roughnesses_scaled, -1, 1)
    slds_scaled = np.clip(slds_scaled, -1, 1)
    background_scaled = np.clip(background_scaled, -1, 1)

    params_scaled = np.concatenate([
        thicknesses_scaled,
        roughnesses_scaled,
        slds_scaled,
        background_scaled
    ], axis=1)

    params = np.concatenate([
        thickness_array,
        roughness_array,
        slds_array,
        background_array
    ], axis=1)
    print(params_scaled.shape)

    params_scaled = torch.tensor(params_scaled, dtype=torch.float32)
    epsilon = torch.from_numpy(np.random.uniform(low=0.1, high=0.3, size=params_scaled.shape)).float()
    epsilon = epsilon.to(params_scaled.device)

    min_bounds_scaled = torch.clamp(params_scaled - epsilon, -1.0, 1.0)
    max_bounds_scaled = torch.clamp(params_scaled + epsilon, -1.0, 1.0)

    Layer_count = len(thickness_array[0]) 
    thickness_min_torch = torch.full((Layer_count, ), thickness_min)
    thickness_max_torch = torch.full((Layer_count, ), thickness_max)

    roughness_min_torch = torch.full((Layer_count+1, ), roughness_min)
    roughness_max_torch = torch.full((Layer_count+1, ), roughness_max)

    sld_min_torch = torch.full((Layer_count+1, ), sld_min)
    sld_max_torch = torch.full((Layer_count+1, ), sld_max)

    background_min_torch = torch.full((1, ), background_min)
    background_max_torch = torch.full((1, ), background_max)

    lower_bounds = torch.cat([thickness_min_torch, roughness_min_torch, sld_min_torch,background_min_torch], dim=0)
    upper_bounds = torch.cat([thickness_max_torch, roughness_max_torch,sld_max_torch,background_max_torch], dim=0)

    span = upper_bounds - lower_bounds
    bounds = (span,lower_bounds)
                             
    scaled_bounds = torch.cat([min_bounds_scaled, max_bounds_scaled], dim=1)

    return params_scaled, params, scaled_bounds, bounds


def scaling_data(curves,q_array,thickness_array, roughness_array, slds_array,background_array):
    from reflectorch.data_generation.scale_curves import LogAffineCurvesScaler

    curve_scaler = LogAffineCurvesScaler(weight=0.2, bias=1.0)
    scaled_curves = curve_scaler.scale(torch.tensor(curves, dtype=torch.float32))

    params_scaled, params,scaled_bounds , bounds = scale_params(thickness_array,roughness_array,slds_array,background_array)

    q_min = q_array.min()
    q_max = q_array.max()
    q_scaled = (q_array - q_min) / (q_max - q_min)
    q_scaled = 2.0 * (q_scaled - 0.5)

    return scaled_curves,q_scaled, params_scaled, params,scaled_bounds ,bounds

def load_scale_data(folder_path, train_split = 0.95):

    print(folder_path)

    with open(folder_path, "rb") as f:
        data_file = pickle.load(f)

    print(data_file.columns)
    print(data_file.iloc[0])

    curves = []
    thickness_array = []
    q_values = []
    roughness_array = []
    slds_array = []
    background_array = []

    for _, row in data_file.iterrows():
        q = np.array(row['Q (Å-1)']).squeeze(-1)  
        intensity = np.array(row['Intensity']).squeeze(-1)  
        thicknesses = np.array(row['Thicknesses (Å)'])[1:]
        slds = np.array(row['SLDs (Å-2)'])
        slds = slds.real.astype(np.float32)
        slds = slds * 1e6
        roughnesses = np.array(row['Roughnesses (Å)'])
        background = np.array(row['Background'])

        curves.append(intensity)
        q_values.append(q)
        slds_array.append(slds)
        roughness_array.append(roughnesses)
        thickness_array.append(thicknesses)
        background_array.append(background)

    curves = np.stack(curves)
    thickness_array = np.array(thickness_array)
    q_values = np.stack(q_values)
    slds_array = np.array(slds_array)
    roughness_array = np.array(roughness_array)
    background_array = np.array(background_array)
    background_array = background_array[:, np.newaxis]

    thickness_array_flipped = thickness_array[::-1] 
    #q_values_flipped = q_values[::-1] 
    slds_array_flipped = slds_array[::-1] 
    roughness_array_flipped = roughness_array[::-1]  

    index = np.arange(len(curves))
    training_indexes, testing_indexes = train_test_split(
        index,
        train_size=train_split,
        shuffle=True,
        random_state=42
    )

    train_data = {
        'curves':    curves[training_indexes],
        'q_values':  q_values[training_indexes],
        'thickness_array': thickness_array_flipped[training_indexes],
        'roughness_array': roughness_array_flipped[training_indexes],
        'slds_array':      slds_array_flipped[training_indexes],
        'background_array': background_array[training_indexes],
    }
    test_data = {
        'curves':    curves[testing_indexes],
        'q_values':  q_values[testing_indexes],
        'thickness_array': thickness_array_flipped[testing_indexes],
        'roughness_array': roughness_array_flipped[testing_indexes],
        'slds_array':      slds_array_flipped[testing_indexes],
        'background_array': background_array[testing_indexes],
    }

    training_num_layers = thicknesses.shape[0]
    testing_num_layers = thicknesses.shape[0]

    training_scaled_curves, training_q_scaled, training_params_scaled, training_params,training_scaled_bounds ,training_bounds = scaling_data(train_data["curves"],train_data["q_values"],train_data["thickness_array"], train_data["roughness_array"], train_data["slds_array"],train_data["background_array"])
    testing_scaled_curves, testing_q_scaled, testing_params_scaled, testing_params,testing_scaled_bounds ,testing_bounds = scaling_data(test_data["curves"],test_data["q_values"],test_data["thickness_array"], test_data["roughness_array"], test_data["slds_array"],test_data['background_array'])
    
    training_data = (train_data["curves"], training_scaled_curves, training_params, training_params_scaled, train_data["q_values"], training_q_scaled, training_num_layers,training_scaled_bounds , training_bounds)
    testing_data = (test_data["curves"], testing_scaled_curves, testing_params, testing_params_scaled,test_data["q_values"], testing_q_scaled, testing_num_layers,testing_scaled_bounds , testing_bounds)

    return training_data,testing_data



