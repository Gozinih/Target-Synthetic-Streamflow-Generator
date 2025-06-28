# c1_ExposureSpacePlot.py
# Visualizes exposure space (mean vs SD changes) for selected locations and months
# using synthetic optimization results stored in .h5 files.

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ====== USER INPUTS ======
locations = [2, 12]                     # Two location number for plotting.      
scenariofolder = "Scenarios"       # The folder of results     
# ========= End ============

# ============================================= Plotting Code ========================================
locations = [l - 1 for l in locations]  # Convert to 0-based for Python
script_dir = os.path.dirname(os.path.abspath(__file__))
scenariofolderpass = os.path.join(script_dir, "..", scenariofolder)  # one level up
scenariofolderpass = os.path.abspath(scenariofolderpass)  # normalize
n_months = 12
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# === Load seasonality & deviation ranges ===
infeasible_path = os.path.join(scenariofolderpass, "OutputData", "Desired_Scenarios_InfeasibleRemoved.h5")
with h5py.File(infeasible_path, "r") as f:
    meanseasonality_change = f["Desired_Sesonality_Mean[Location x Month]"][:]
    SDseasonality_change = f["Desired_Sesonality_SD[Location x Month]"][:]
    desired_scenarios1 = f["Desired_Deviations[Scenario x 2]"][:]
    desired_scenarios_monthly = f["Desired_Deviations_Monthly[Scenario x 24]"][:]

# === Target deviation zone (rectangle bounds) ===
rect_x_min, rect_x_max = desired_scenarios1[:, 0].min(), desired_scenarios1[:, 0].max()
rect_y_min, rect_y_max = desired_scenarios1[:, 1].min(), desired_scenarios1[:, 1].max()

n_scenarios = desired_scenarios1.shape[0]
n_locs = len(locations)

# === Allocate storage ===
x_opt_all = np.zeros((n_scenarios, n_months, n_locs))
y_opt_all = np.zeros((n_scenarios, n_months, n_locs))

# === Load optimized mean and SD deviation per scenario ===
for sce in range(n_scenarios):
    file_path = os.path.join(scenariofolderpass, "OutputData", f"Scenario{sce+1}.h5")
    if os.path.exists(file_path):
        with h5py.File(file_path, "r") as f:
            mean_dev = f["Opt_Mean_Deviation[Location x Month]"][:]
            sd_dev = f["Opt_SD_Deviation[Location x Month]"][:]
            for j, loc in enumerate(locations):
                x_opt_all[sce, :, j] = mean_dev[loc, :]
                y_opt_all[sce, :, j] = sd_dev[loc, :]

# === Plot Setup ===
fig, axes = plt.subplots(n_locs * 2, 6, figsize=(18, 6 * n_locs))
axes = axes.flatten()

for plot_idx, loc in enumerate(locations):
    for month in range(n_months):
        ax = axes[plot_idx * 12 + month]

        # Plot optimized points
        ax.scatter(x_opt_all[:, month, plot_idx],
                   y_opt_all[:, month, plot_idx],
                   s=10, color='blue', label="Synthetic Scenarios")

        # Seasonality origin shift
        x_origin = meanseasonality_change[loc, month]
        y_origin = SDseasonality_change[loc, month]
        ax.plot(x_origin, y_origin, 'ko', label="Target Seasonality")
        ax.plot(0, 0, 'ro', label="(0,0) Origin")

        # Target deviation rectangle, adjusted with origin shift
        rect_x = np.array([rect_x_min, rect_x_max, rect_x_max, rect_x_min])
        rect_y = np.array([rect_y_min, rect_y_min, rect_y_max, rect_y_max])
        rect_x = rect_x + x_origin + 0.01 * rect_x * x_origin
        rect_y = rect_y + y_origin + 0.01 * rect_y * y_origin
        ax.fill(rect_x, rect_y, color='#5dade2', alpha=0.4, edgecolor='b', label="Target Deviation")

        ax.set_title(f"Loc {loc+1} - {month_names[month]}",fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=6)

# === Shared legend and universal axis titles ===
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4, fontsize=14, prop={'weight': 'bold'})
fig.text(0.5, 0.005, 'Resultant Monthly Mean Change (%)', ha='center', fontsize=14, fontweight='bold')
fig.text(0.01, 0.5, 'Resultant Monthly SD Change (%)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

# === Layout adjustment to leave space for text and legend ===
fig.tight_layout(rect=[0.015, 0.03, 1, 0.95])
plt.show()