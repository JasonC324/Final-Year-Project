import numpy as np
import matplotlib.pyplot as plt
import torch
# --- Example inputs (replace these with your actual values) ---
d_full_rel_max = 90  # must be divisible by 3

if d_full_rel_max % 3 != 0:
    raise ValueError(f"Number {d_full_rel_max} is not divisible by 3.")

n = d_full_rel_max
total_layers = n
total_repeats = n // 3  # integer division

# Fake parametrized_model.T unpacked values (replace these with your actual arrays)
# For demo, just use some example scalar or 1D arrays with batch size 1
batch_size = 1

# Each param: shape (batch_size,)
d_full_rel = np.array([15.0])
rel_sigmas = np.array([0.1])
d_A = np.array([10.0])
d_B = np.array([12.0])
d_C = np.array([14.0])
s_A = np.array([0.8])
s_B = np.array([0.9])
s_C = np.array([1.0])
r_A = np.array([2.0])
r_B = np.array([3.0])
r_C = np.array([4.0])
SLD_diff_1_2 = np.array([1.0])
SLD_diff_2_3 = np.array([1.5])
d_sio2 = np.array([5.0])
s_sio2 = np.array([0.7])
s_si = np.array([0.6])
r_sio2 = np.array([2.2])
r_si = np.array([3.3])
dr_sigmoid_pos_AB = np.array([0.2])
dr_sigmoid_width_AB = np.array([0.3])
dr_sigmoid_pos_BC = np.array([0.4])
dr_sigmoid_width_BC = np.array([0.5])

# --- Helper sigmoid ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Calculate positions ---
r_positions = total_layers - np.arange(total_layers)  # shape (total_layers,)

# For batch computation, add batch dimension at front
# We tile for batch_size=1 here, but you can extend this for batches:
r_positions = np.tile(r_positions, (batch_size, 1))  # shape (batch_size, total_layers)
print(r_positions)

# Add axis for broadcasting batch parameters
d_full_rel = d_full_rel[:, None]            # (batch_size, 1)
rel_sigmas = rel_sigmas[:, None]            # (batch_size, 1)
dr_sigmoid_pos_AB = dr_sigmoid_pos_AB[:, None]
dr_sigmoid_width_AB = dr_sigmoid_width_AB[:, None]
dr_sigmoid_pos_BC = dr_sigmoid_pos_BC[:, None]
dr_sigmoid_width_BC = dr_sigmoid_width_BC[:, None]
SLD_diff_1_2 = SLD_diff_1_2[:, None]
SLD_diff_2_3 = SLD_diff_2_3[:, None]

r_A = r_A[:, None]  # (batch_size, 1)
r_B = r_B[:, None]
r_C = r_C[:, None]

# --- Calculate modulation sigmoids ---
r_modulations = sigmoid(
    -(r_positions - 3 * d_full_rel) / rel_sigmas
)  # shape (batch_size, total_layers)

idx_B = np.arange(1, total_layers, 3)
pos_B = r_positions[:, idx_B]

idx_C = np.arange(2, total_layers, 3)
pos_C = r_positions[:, idx_C]

mod_AB = 1 - sigmoid(
    -(pos_B - 3 * d_full_rel + 2 * dr_sigmoid_pos_AB) / dr_sigmoid_width_AB
)

mod_BC = 1 - sigmoid(
    -(pos_C - 3 * d_full_rel + 2 * dr_sigmoid_pos_BC) / dr_sigmoid_width_BC
)

# --- Repeat r_A, r_B, r_C for all repeats ---
r_A_rep = np.repeat(r_A, total_repeats, axis=1)  # shape (batch_size, total_repeats)
r_B_rep = np.repeat(r_B, total_repeats, axis=1)
r_C_rep = np.repeat(r_C, total_repeats, axis=1)

# --- Calculate modulated SLD for B and C layers ---
modulated_B = r_A_rep + SLD_diff_1_2 * mod_AB
modulated_C = r_B_rep + SLD_diff_2_3 * mod_BC
modulated_A = r_A_rep

# --- Concatenate A, B, C layers ---
slds_modulated = np.concatenate([modulated_A, modulated_B, modulated_C], axis=1)  # (batch_size, n)

# --- Apply overall modulation ---
slds_final = slds_modulated * r_modulations

slds_final = np.stack([r_A_rep, r_B_rep, r_C_rep], axis=2).reshape(batch_size, -1)

i = 5
k = 0.3       # steepness
z0 = 3* d_full_rel  # midpoint halfway into layer

center_sld = np.mean(slds_final)

squeeze = 1 - 1 / (1 + np.exp(-k * (r_positions - z0)))
slds_squeezed = center_sld + squeeze * (slds_final - center_sld)


r_modulations = sigmoid(-(r_positions - 3 * d_full_rel) / i)
    

slds_moded = slds_squeezed * r_modulations

# --- Repeat thicknesses and roughness for each repeat ---
d_A_stack = np.repeat(d_A[:, None], total_repeats, axis=1)
d_B_stack = np.repeat(d_B[:, None], total_repeats, axis=1)
d_C_stack = np.repeat(d_C[:, None], total_repeats, axis=1)
thickness_abc = np.stack([d_A_stack, d_B_stack, d_C_stack], axis=2).reshape(batch_size, -1)

s_A_stack = np.repeat(s_A[:, None], total_repeats, axis=1)
s_B_stack = np.repeat(s_B[:, None], total_repeats, axis=1)
s_C_stack = np.repeat(s_C[:, None], total_repeats, axis=1)
roughness_abc = np.stack([s_A_stack, s_B_stack, s_C_stack], axis=2).reshape(batch_size, -1)

# --- Add SiO2 cap thickness and roughness, substrate roughness ---
thicknesses = np.concatenate([thickness_abc, d_sio2[:, None]], axis=1)
roughnesses = np.concatenate([roughness_abc, s_sio2[:, None], s_si[:, None]], axis=1)
slds = np.concatenate([slds_final, r_sio2[:, None], r_si[:, None]], axis=1)

# --- Final params dict ---
params = dict(
    thickness=thicknesses,
    roughness=roughnesses,
    sld=slds
)

# --- Example print ---
print("Thicknesses:", params['thickness'])
print("Roughnesses:", params['roughness'])
print("SLDs:", params['sld'])


# Compute depths for plotting (center of layers)
depths = np.arange(0, len(params['sld'][0]), 1)

print(depths)
print(r_positions)

# === PLOT ===

fig, ax1 = plt.subplots()

ax1.plot(depths,params['sld'][0], marker='o', label='Not Modulated SLD')
ax1.plot(depths[:-2],slds_squeezed[0], marker='x', linestyle='--', label='SLD Squeezed')
ax1.plot(depths[:-2],slds_moded[0], marker='x', linestyle='--', label='10')
ax1.invert_yaxis()
ax1.set_xlabel('Depth (nm)')
ax1.set_ylabel('SLD (10⁻⁶ Å⁻²)')
ax1.set_title('Layer SLD Profile with Sigmoid Modulations')

# Show thickness boundaries as horizontal lines
layer_boundaries = np.cumsum(thickness_abc)
for boundary in layer_boundaries:
    ax1.axhline(boundary, color='gray', linestyle=':', alpha=0.5)

ax1.legend()
ax1.grid(True)
plt.ylim(0, 5)  # Set x-axis limits to focus around your SLD range


plt.show()
