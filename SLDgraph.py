import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import erf
import re

# === [STEP 1] Define Constants ===
NA = 6.022e23           # Avogadro's number [mol^-1]
e_radius = 2.818e-13          # Classical electron radius [cm]
cm_to_angstrom = 1e8    # Conversion from cm^-2 to Å^-2
x_ray_wavelen = 1.54
beta = 0.0

input_angle_ray = 4
angle_step_size = 0.2

Gaussian_roughness = 1.0

# === [STEP 2] Define Layer Materials with Properties ===
# Each layer is defined by: thickness [nm], density [g/cm^3], molar mass [g/mol], atomic number Z


layers_string = "[SiO2/5/2.2/60.08/30,Al/10/2.7/26.98/13]"


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
                if len(properties) != 5:
                    raise ValueError(f"Layer format invalid: '{content}'")

                name, thickness, density, molar_mass, Z = properties
                layer_dict = {
                    "name": name,
                    "thickness": float(thickness),
                    "density": float(density),
                    "molar_mass": float(molar_mass),
                    "Z": int(Z)
                }
                layers.append(layer_dict)
    return layers

# === [STEP 3] Function to Compute SLD for a Material ===
def compute_sld(density, molar_mass, Z):
    number_density = (NA * density) / molar_mass  # atoms/cm^3
    b = e_radius * Z                                     # scattering length [cm]
    sld_cm = number_density * b                    # SLD in cm^-2
    sld_angstrom = sld_cm / cm_to_angstrom         # Convert to Å^-2
    return sld_angstrom

# === [STEP 4] Generate SLD Profile with Depth ===
def make_gaussian_sld_profile(slds ,thicknesses, resolution=0.1, sigma = Gaussian_roughness):
    total_depth = sum(thicknesses)
    z = np.arange(0, total_depth, resolution)
    sld_profile = np.zeros_like(z)

    interfaces = np.cumsum(thicknesses)
    interfaces = np.insert(interfaces, 0, 0)

    for i in range(1,len(slds)):
        interface = interfaces[i]
        p1 = slds[i-1]
        p2 = slds[i]

        erf_transition = 0.5 * (1 + erf((z - interface) / (np.sqrt(2) * sigma)))
        sld_profile += (p2 - p1) * erf_transition
    
    return z, sld_profile

def compute_refractive_index(sld):
        delta = (e_radius*1e10 * x_ray_wavelen**2 * sld) / 2 * math.pi
        n_layer = 1- delta + (1j* beta)
        return n_layer, delta

def parrat_reflectivity(depth_list, sld_list,input_angle_ray, sigma = Gaussian_roughness):
    n_list_with_delta = [compute_refractive_index(sld) for sld in sld_list]
    reflectivity_coe_list = []
    for theta in np.arange(0,input_angle_ray+angle_step_size,angle_step_size):

        k_vec = 2*(math.pi) / x_ray_wavelen
        kz = k_vec*math.sin(theta)
        q = 2*kz

        delta = n_list_with_delta[-1][1]
        qj_pre = np.sqrt(q**2 - 8 * (k_vec**2) * delta + 8 * 1j * (k_vec**2) * beta)
        kz_pre = np.sqrt(kz**2 - 2 * delta * k_vec**2 + 2 * 1j * beta * k_vec**2)
        rj_pre = 0
        for sld in reversed(range(len(sld_list) - 1)):

            delta = n_list_with_delta[sld][1]
            q_layer = np.sqrt(q**2 - 8 * (k_vec**2) * delta + 8 * 1j * (k_vec**2) * beta)
            kz_layer = np.sqrt(kz**2 - 2 * delta * k_vec**2 + 2 * 1j * beta * k_vec**2)
            rj_prime = ((qj_pre - q_layer) / (qj_pre + q_layer)) * np.exp(2 * sigma**2 * kz_layer * kz_pre)
            
            if sld == (len(sld_list) - 2):
                rj = rj_prime
            else:
                temp = np.exp(2j * kz_layer * depth_list[sld])
                rj = (rj_prime + rj_pre * temp) / (1 + rj_prime * rj_pre * temp)
            
            rj_pre = rj
            qj_pre = q_layer
            kz_pre = kz_layer
        final_reflectivity_coe = np.abs(rj**2)
        reflectivity_coe_list.append(final_reflectivity_coe)
    
    return reflectivity_coe_list

# === [STEP 5] Compute and Plot the SLD Profile ===
layers = make_layer_list(layers_string)
top_layer = {"name": "air/water", "thickness": 5, "density": 0, "molar_mass": 1.0, "Z": 0}

slds = []
thicknesses= []

air_sld = compute_sld(top_layer["density"], top_layer["molar_mass"], top_layer["Z"])
air_thicknesses = top_layer["thickness"]
slds.append(air_sld)
thicknesses.append(air_thicknesses)

layer_sld = [compute_sld(l["density"], l["molar_mass"], l["Z"]) for l in layers]
layer_thicknesses = [l["thickness"] for l in layers]
slds.extend(layer_sld)
thicknesses.extend(layer_thicknesses)

substrate_layer = {"name": "substrate", "thickness": 5, "density": 0, "molar_mass": 1.0, "Z": 0}
substrate_sld = compute_sld(substrate_layer["density"], substrate_layer["molar_mass"], substrate_layer["Z"])
substrate_thicknesses = substrate_layer["thickness"]
slds.append(substrate_sld)
thicknesses.append(substrate_thicknesses)


depth, sld = make_gaussian_sld_profile(slds,thicknesses)

plt.figure(figsize=(6, 4))
plt.plot(depth,sld, label="SLD Profile")  # scaled for readable plot
plt.xlabel("z (Å)")
plt.ylabel("SLD [Å$^{-2}$]")
plt.title("Scattering Length Density (SLD) Profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()

k_vec = 2*(math.pi) / x_ray_wavelen
# === [STEP 6] Curve Graph ===
reflectivity_coe_list = parrat_reflectivity(depth, sld, input_angle_ray*2)

plt.figure(figsize=(6, 4))
plt.plot(np.arange(0,input_angle_ray*2+angle_step_size,angle_step_size),reflectivity_coe_list, label="Reflectivity curve")  # scaled for readable plot
plt.xlabel("2θ (*)")
plt.ylabel("R(a.u.)")
plt.title("Scattering Length Density (SLD) Profile")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()