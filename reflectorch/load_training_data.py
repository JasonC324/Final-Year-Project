import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def scale_params(thickness_array,roughness_array,slds_array):
    roughness_min = 0.0
    roughness_max = 60.0

    sld_min = slds_array.min()
    sld_max = slds_array.max()

    thickness_min, thickness_max = 0.0, 500.0

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

    thicknesses_scaled = np.clip(thicknesses_scaled, -1, 1)
    roughnesses_scaled = np.clip(roughnesses_scaled, -1, 1)
    slds_scaled = np.clip(slds_scaled, -1, 1)

    params_scaled = np.concatenate([
        thicknesses_scaled,
        roughnesses_scaled,
        slds_scaled
    ], axis=1)

    params = np.concatenate([
        thickness_array,
        roughness_array,
        slds_array
    ], axis=1)
    print(params_scaled.shape)

    Layer_count = len(thickness_array[0]) 
    thickness_min_torch = torch.full((Layer_count, ), thickness_min)
    thickness_max_torch = torch.full((Layer_count, ), thickness_max)

    roughness_min_torch = torch.full((Layer_count+1, ), roughness_min)
    roughness_max_torch = torch.full((Layer_count+1, ), roughness_max)

    sld_min_torch = torch.full((Layer_count+1, ), sld_min)
    sld_max_torch = torch.full((Layer_count+1, ), sld_max)

    lower_bounds = torch.cat([thickness_min_torch, roughness_min_torch, sld_min_torch], dim=0)
    upper_bounds = torch.cat([thickness_max_torch, roughness_max_torch,sld_max_torch], dim=0)

    span = upper_bounds - lower_bounds
    bounds = (span,lower_bounds)
    
    lb = lower_bounds.real.float()
    ub = upper_bounds.real.float()
    span = ub - lb
    lb_scaled =  2 * (lb - lb) / span - 1
    ub_scaled =  2 * (ub - lb) / span - 1
                             
    scaled_bounds = torch.cat([lb_scaled, ub_scaled], dim=0)

    return params_scaled, params, scaled_bounds, bounds


def scaling_data(curves,thicknesses,thickness_array, roughness_array, slds_array):
    from reflectorch.data_generation.scale_curves import LogAffineCurvesScaler

    curve_scaler = LogAffineCurvesScaler()
    scaled_curves = curve_scaler.scale(torch.tensor(curves, dtype=torch.float32))

    num_layers = thicknesses.shape[0]
    params_scaled, params,scaled_bounds , bounds = scale_params(thickness_array,roughness_array,slds_array)

    return scaled_curves, num_layers, params_scaled, params,scaled_bounds ,bounds

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

    for _, row in data_file.iterrows():
        q = np.array(row['Q (Å-1)']).squeeze(-1)  
        intensity = np.array(row['Intensity']).squeeze(-1)  
        thicknesses = np.array(row['Thicknesses (Å)'])[1:]
        slds = np.array(row['SLDs (Å-2)'])
        slds = slds.real.astype(np.float32)
        roughnesses = np.array(row['Roughnesses (Å)'])

        curves.append(intensity)
        q_values.append(q)
        slds_array.append(slds)
        roughness_array.append(roughnesses)
        thickness_array.append(thicknesses)

    curves = np.stack(curves)
    thickness_array = np.array(thickness_array)
    q_values = np.stack(q_values)
    slds_array = np.array(slds_array)
    roughness_array = np.array(roughness_array)

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
        'thickness_array': thickness_array[training_indexes],
        'roughness_array': roughness_array[training_indexes],
        'slds_array':      slds_array[training_indexes],
    }
    test_data = {
        'curves':    curves[testing_indexes],
        'q_values':  q_values[testing_indexes],
        'thickness_array': thickness_array[testing_indexes],
        'roughness_array': roughness_array[testing_indexes],
        'slds_array':      slds_array[testing_indexes],
    }

    training_scaled_curves, training_num_layers, training_params_scaled, training_params,training_scaled_bounds ,training_bounds = scaling_data(train_data["curves"],thicknesses,train_data["thickness_array"], train_data["roughness_array"], train_data["slds_array"])
    testing_scaled_curves, testing_num_layers, testing_params_scaled, testing_params,testing_scaled_bounds ,testing_bounds = scaling_data(test_data["curves"],thicknesses,test_data["thickness_array"], test_data["roughness_array"], test_data["slds_array"])
    
    training_data = (train_data["curves"], training_scaled_curves, training_params, training_params_scaled, train_data["q_values"], training_num_layers,training_scaled_bounds , training_bounds)
    testing_data = (test_data["curves"], testing_scaled_curves, testing_params, testing_params_scaled, test_data["q_values"], testing_num_layers,testing_scaled_bounds , testing_bounds)

    return training_data,testing_data


if __name__ == '__main__':
    curves, scaled_curves, params, params_scaled, q_values, num_layers,scaled_bounds , bounds = load_scale_data(
        "./Ta_Pt_bilayer_data_1000/Ta_Pt_BL_training_data_1000_pickle"
    )

    print("scaled_curves:", scaled_curves.shape)
    print("scaled_params:", params_scaled.shape)
    print("q_values:     ", q_values.shape)
    print("bounds:     ", scaled_bounds.shape)




