# c3_TimeSeries_MonthlyFlow_4Locations.py
# Plots time series of monthly flow: Synthetic vs Historical (for 4 specific locations)
# Each subplot overlays the recorded time series with a min-max envelope from all synthetic scenarios
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

script_dir = os.path.dirname(os.path.abspath(__file__))
scenariofolderpass = os.path.abspath(os.path.join(script_dir, "..", scenariofolder))
output_path = os.path.join(scenariofolderpass, "OutputData")
inputdata_path = os.path.join(scenariofolderpass, "InputData", "Inputs.h5")
infeasible_path = os.path.join(output_path, "Desired_Scenarios_InfeasibleRemoved.h5")

# Load scenario metadata
with h5py.File(infeasible_path, "r") as f:
    desired_scenarios1 = f["Desired_Deviations[Scenario x 2]"][:]
n_scenarios = desired_scenarios1.shape[0]
with h5py.File(inputdata_path, "r") as f:
    numberofyears_syntheticdata = int(f["Metadata"].attrs["numberofyears_syntheticdata+1"]) - 1
N = numberofyears_syntheticdata * 12

# Allocate data arrays
inputdata_recorded = np.zeros((N, n_locs))
synthetic_data = np.zeros((N, n_scenarios, n_locs))

# Load synthetic and recorded monthly flow data from .h5
for sce in range(n_scenarios):
    file_path = os.path.join(output_path, f"Scenario{sce+1}.h5")
    if os.path.exists(file_path):
        with h5py.File(file_path, "r") as f:
            Monthly_Synthetic = f["Monthly_Synthetic(million m3/month)[Year x Month x Location]"][:]
            Monthly_Recorded = f["Monthly_Recorded(million m3/month)[Year x Month x Location]"][:]
            for j, k in enumerate(locations):
                synthetic_data[:, sce, j] = Monthly_Synthetic[:, :, k].T.flatten()
                if sce == 0:
                    inputdata_recorded[:, j] = Monthly_Recorded[:, :, k].T.flatten()

# Plotting
nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
axes = axes.flatten()
for j in range(n_locs):
    ax = axes[j]
    line_rec, = ax.plot(inputdata_recorded[:, j], 'k', linewidth=0.3)
    synthetic_min = np.min(synthetic_data[:, :, j], axis=1)
    synthetic_max = np.max(synthetic_data[:, :, j], axis=1)
    x = np.arange(N)
    fill_syn = ax.fill_between(x, synthetic_min, synthetic_max, color='blue', alpha=0.3)
    ax.set_xlim([0, N])
    ax.set_xticks(np.arange(0, N + 1, 50))
    ax.tick_params(labelsize=10)
    ax.set_title(f"Loc {locations[j]+1}", fontsize=14)

# Shared legend and labels
custom_lines = [
   Line2D([0], [0], color='blue', lw=6, alpha=0.3, label='Synthetic Scenarios'),
   Line2D([0], [0], color='black', lw=2, label='Recorded')
]
fig.legend(custom_lines, ['Synthetic Scenarios', 'Recorded'],
           loc='lower right', fontsize=14, frameon=False)

fig.text(0.5, 0.04, 'Month', ha='center', fontsize=14, fontweight='bold')
fig.text(0.05, 0.5, 'Total Monthly Flow (m³/s)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
plt.show()
