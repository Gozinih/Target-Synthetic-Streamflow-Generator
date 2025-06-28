# c2_FlowDurationCurve_4Locations.py
# Monthly Flow Duration Curve: Synthetic vs Historical (for 4 specific locations)
# Each subplot compares synthetic scenario FDC envelopes to historical records
# Uses synthetic data stored in .h5 files under Scenarios/OutputData/

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ====== USER INPUTS ======
locations = [2, 4, 6, 12]            # Four location number for plotting  
scenariofolder = "Scenarios"         # The folder of results  
# ========= End ============

# ============================================= Plotting Code ========================================
locations = [l - 1 for l in locations]
n_locs = len(locations)

# === Setup Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
scenariofolderpass = os.path.abspath(os.path.join(script_dir, "..", scenariofolder))
output_path = os.path.join(scenariofolderpass, "OutputData")
inputdata_path = os.path.join(scenariofolderpass, "InputData", "Inputs.h5")
infeasible_path = os.path.join(output_path, "Desired_Scenarios_InfeasibleRemoved.h5")

# === Load number of years from metadata ===
with h5py.File(inputdata_path, "r") as f:
    numberofyears_syntheticdata = int(f["Metadata"].attrs["numberofyears_syntheticdata+1"]) - 1

# === Load scenario metadata ===
with h5py.File(infeasible_path, "r") as f:
    desired_scenarios1 = f["Desired_Deviations[Scenario x 2]"][:]
n_scenarios = desired_scenarios1.shape[0]

N = numberofyears_syntheticdata * 12
prob_exceedance = np.arange(1, N + 1) / (N + 1)

# === Allocate data arrays ===
inputdata_recorded = np.zeros((N, n_locs))
synthetic_data = np.zeros((N, n_scenarios, n_locs))

# === Load synthetic and recorded monthly flow data from .h5 ===
for sce in range(n_scenarios):
    file_path = os.path.join(output_path, f"Scenario{sce+1}.h5")
    if os.path.exists(file_path):
        with h5py.File(file_path, "r") as f:
            Monthly_Synthetic = f["Monthly_Synthetic[Year x Month x Location]"][:]
            Monthly_Recorded = f["Monthly_Recorded[Year x Month x Location]"][:]
            for j, k in enumerate(locations):
                synthetic_data[:, sce, j] = Monthly_Synthetic[:, :, k].T.flatten()
                if sce == 0:
                    inputdata_recorded[:, j] = Monthly_Recorded[:, :, k].T.flatten()

# === Compute FDCs ===
fdc_recorded = np.sort(inputdata_recorded, axis=0)[::-1, :]
fdc_synthetic = np.sort(synthetic_data, axis=0)[::-1, :, :]

# === Plotting ===
fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
axes = axes.flatten()

for j in range(n_locs):
    ax = axes[j]

    # Plot synthetic scenarios (with positive-only values)
    for sce in range(n_scenarios):
        y_vals = fdc_synthetic[:, sce, j]
        y_vals = y_vals[y_vals > 0]
        x_vals = prob_exceedance[:len(y_vals)] * 100
        ax.plot(x_vals, y_vals, color=(0.0, 0.3, 1.0), linewidth=1)

    # Plot historical
    y_hist = fdc_recorded[:, j]
    y_hist = y_hist[y_hist > 0]
    x_hist = prob_exceedance[:len(y_hist)] * 100
    ax.plot(x_hist, y_hist, color='black', linewidth=2)

    ax.set_yscale('log')
    ax.set_title(f"Loc {locations[j]+1}", fontsize=14)
    ax.grid(True)
    ax.tick_params(labelsize=10)

# Shared legend
custom_lines = [
   Line2D([0], [0], color=(0.0, 0.3, 1.0), lw=1, label='Synthetic Scenarios'),
   Line2D([0], [0], color='black', lw=2, label='Historical')
]
fig.legend(custom_lines, ['Synthetic Scenarios', 'Historical'],
           loc='lower right', fontsize=14, frameon=False)

fig.text(0.5, 0.04, 'Exceedance Probability (%)', ha='center', fontsize=14, fontweight='bold')
fig.text(0.05, 0.5, 'Total Monthly Flow (m^/s, log scale)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
plt.show()
