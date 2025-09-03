import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import erf
import re

NA = 6.022e23           # Avogadro's number [mol^-1]
e_radius = 2.818e-5          # Classical electron radius [A]
x_ray_wavelen = 1.54e-10     #  [M]
c = 3e8                      #  [M/s]
h = 6.626e-34

input_angle_ray = 4
angle_step_size = 0.02

Gaussian_roughness = 1.0

#Each layer is defined by: thickness [nm], density [g/cm^3], molar mass [g/mol]
layers_string = "[SiO2/5/2.65/60.08,Al/10/2.7/26.98]3"

def get_molecule(formula):
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)

    parsed = []
    for element, count in matches:
        count = int(count) if count else 1
        parsed.append((element, count))
    return parsed

def load_scattering_data(atom_name,energy_need):
    min_diff = 1e10
    atom_name = atom_name.lower()
    with open(f"sf/{atom_name}.nff", "r") as file:
        for line in file:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = line.split()
            try:
                energy = float(parts[0])
                f1 = float(parts[1])
                f2 = float(parts[2])
                diff = abs(energy - energy_need)
                if diff < min_diff:
                    min_diff = diff
                    best_match = (energy, f1, f2)
            except (ValueError, IndexError):
                continue
        if best_match:
            energy, f1, f2 = best_match
            return f1, f2
        else:
            raise ValueError("No valid data found in file.")

def make_layer_list(layers_string):
    if not isinstance(layers_string, str):
        layers_string = str(layers_string)
    pattern = r"\[([^\[\]]+)\](\d*)"
    matches = re.findall(pattern, layers_string)

    layers = []
    for content, repeat_str in matches:
        repeat_layers = content.split(',')
        repeat = int(repeat_str) if repeat_str else 1
        for _ in range(repeat):
            for layer in repeat_layers:
                properties = layer.split('/')
                if len(properties) != 4:
                    raise ValueError(f"Layer format invalid: '{content}'")
                name, thickness, density, molar_mass = properties
                
                atoms = get_molecule(name)
                eV = h*c/x_ray_wavelen
                eV = eV * 6.242e18

                total_atoms = 0
                f1_total = 0
                f2_total = 0
                for atom in atoms:
                    f1, f2 = load_scattering_data(atom[0],eV)
                    total_atoms = total_atoms + atom[1]
                    f1_total = f1 * atom[1] + f1_total
                    f2_total = f2 * atom[1] + f2_total
                f1_avg = f1_total/ total_atoms
                f2_avg = f2_total / total_atoms
                
                layer_dict = {
                    "name": name,
                    "thickness": float(thickness)*10,
                    "density": float(density),
                    "molar_mass": float(molar_mass),
                    "f1": float(f1_avg),
                    "f2": float(f2_avg)
                }
                layers.append(layer_dict)
    return layers

def compute_sld(density, molar_mass, f1 = 0,f2 = 0):
    atom_density = (NA * density) / molar_mass
    atom_density_angstrom = atom_density * 1e-24
    sld = atom_density_angstrom * e_radius * (f1 + 1j * f2)
    return sld.real, sld.imag

def make_gaussian_sld_profile(slds ,thicknesses, resolution=0.1, sigma = Gaussian_roughness):
    total_depth = sum(thicknesses)
    z = np.arange(0, total_depth, resolution)
    sld_profile_real = np.zeros_like(z)
    sld_profile_imag = np.zeros_like(z)

    interfaces = np.cumsum(thicknesses)
    interfaces = np.insert(interfaces, 0, 0)

    for i in range(1,len(slds)):
        interface = interfaces[i]
        p1 = slds[i-1][0]
        p2 = slds[i][0]

        erf_transition = 0.5 * (1 + erf((z - interface) / (np.sqrt(2) * sigma)))
        sld_profile_real += (p2 - p1) * erf_transition

        interface = interfaces[i]
        p1 = slds[i-1][1]
        p2 = slds[i][1]

        erf_transition = 0.5 * (1 + erf((z - interface) / (np.sqrt(2) * sigma)))
        sld_profile_imag += (p2 - p1) * erf_transition
    
    return z, sld_profile_real, sld_profile_imag

def compute_refractive_index(sld,layer):
        b = layer["f1"] + 1j * layer["f2"]
        rho = sld[0] + 1j * sld[1]
        n = 1 - (b * rho * (x_ray_wavelen*1e10)**2) / (2 * math.pi)
        delta = 1 - n.real
        beta = n.imag
        return delta, beta

def parrat_reflectivity(depth_list, sld_list,input_angle_ray, layers, sigma = Gaussian_roughness):
    delta_beta = [compute_refractive_index(sld,layer) for sld, layer in zip(sld_list,layers)]
    reflectivity_coe_list = []
    for theta in np.arange(0+angle_step_size,input_angle_ray+angle_step_size,angle_step_size):

        k_vec = 2*(math.pi) / (x_ray_wavelen*1e10)
        theta_radians = math.radians(theta)
        kz = k_vec*math.sin(theta_radians)
        q = 2*kz
        
        delta = delta_beta[-1][0]
        beta = delta_beta[-1][1]
        qj_pre = np.sqrt(complex(q**2 - 8 * (k_vec**2) * delta + 8 * 1j * (k_vec**2) * beta))
        kz_pre = np.sqrt(complex(kz**2 - 2 * delta * k_vec**2 + 2 * 1j * beta * k_vec**2))
        rj_pre = 0
        for j in reversed(range(len(sld_list) - 1)):
            
            delta = delta_beta[j][0]
            beta = delta_beta[j][1]
            q_layer = np.sqrt(complex(q**2 - 8 * (k_vec**2) * delta + 8 * 1j * (k_vec**2) * beta))
            kz_layer = np.sqrt(complex(kz**2 - 2 * delta * k_vec**2 + 2 * 1j * beta * k_vec**2))
            
            damp_arg = -2 * sigma**2 * np.real(kz_layer * kz_pre)
            damp_arg = np.clip(damp_arg, -700, 0)
            damp_exp = np.exp(damp_arg)
            
            rj_prime = ((qj_pre - q_layer) / (qj_pre + q_layer)) * damp_exp

            if j == (len(sld_list) - 2):
                rj = rj_prime
            else:
                temp = np.exp(2j * kz_layer * depth_list[j])
                rj = (rj_prime + rj_pre * temp) / (1 + rj_prime * rj_pre * temp)
            
            rj_pre = rj
            qj_pre = q_layer
            kz_pre = kz_layer
        final_reflectivity_coe = np.abs(rj)**2
        reflectivity_coe_list.append(final_reflectivity_coe)
    
    return reflectivity_coe_list

layers = make_layer_list(layers_string)
top_layer = {"name": "air/water", "thickness": 5, "density": 0, "molar_mass": 1.0, "f1": 1.0, "f2": 1.0}

slds = []
thicknesses= []

air_sld = compute_sld(top_layer["density"], top_layer["molar_mass"])
air_thicknesses = top_layer["thickness"]
slds.append(air_sld)
thicknesses.append(air_thicknesses)

layer_sld = [compute_sld(l["density"], l["molar_mass"], l["f1"], l["f2"]) for l in layers]
layer_thicknesses = [l["thickness"] for l in layers]
slds.extend(layer_sld)
thicknesses.extend(layer_thicknesses)

substrate_layer = {"name": "substrate", "thickness": 5, "density": 0, "molar_mass": 1.0, "f1": 1.0, "f2": 1.0}
substrate_sld = compute_sld(substrate_layer["density"], substrate_layer["molar_mass"])
substrate_thicknesses = substrate_layer["thickness"]
slds.append(substrate_sld)
thicknesses.append(substrate_thicknesses)

film_layers = [top_layer] + layers + [substrate_layer]


depth, sld_real, sld_imag = make_gaussian_sld_profile(slds,thicknesses)
print(sld_real)
print(sld_imag)

k_vec = 2*(math.pi) / x_ray_wavelen
reflectivity_coe_list=[]
reflectivity_coe_list = parrat_reflectivity(thicknesses, slds, input_angle_ray*2,film_layers)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(depth, sld_real, label="Real", color="blue")
axes[0].plot(depth, sld_imag, label="Imaginary", color="orange")
axes[0].set_xlabel("z (Å)")
axes[0].set_ylabel("SLD [reÅ$^{-2}$]")
axes[0].set_title("SLD Profile")
axes[0].legend()

axes[1].plot(np.arange(0+angle_step_size,input_angle_ray*2+angle_step_size,angle_step_size), reflectivity_coe_list, label="Reflectivity curve")
axes[1].set_xlabel("2θ (degrees)")
axes[1].set_ylabel("Reflectivity R")
axes[1].set_title('XRR Reflectivity (log scale)')
axes[1].set_yscale("log")
axes[1].legend()

plt.tight_layout()
plt.show()