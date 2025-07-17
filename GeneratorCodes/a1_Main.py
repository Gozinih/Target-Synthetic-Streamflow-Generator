# a1_Main.py

import pandas as pd
import numpy as np
import os
import h5py

from a2_MatrixYear import matrix_year
from a3_BoundaryCoordinateGenerator import boundary_coordinate_generator
from a7_RemoveInfeasibleScenarios import remove_infeasible_scenarios
from a9_ModifyInfeasibleScenarios import adjust_scenario_to_feasible
from a10_InverseApproach_and_MonthlytoDaily import perform_inverse_optimization_and_disaggregation

# start
# ====================================================================================
#                             USER-DEFINED INPUT SECTION
# ====================================================================================
# Update the following values and files as needed before running the script

# === Input Excel Files ===
recorded_data = 'RecordedData.xlsx'                    # <- Daily flow data (cms) & isLocal (1 = local, 0 = nonlocal) flags [Sheet1 & Sheet2]
mean_seasonality_data = 'Mean_Seasonality.xlsx'        # <- Seasonal mean change (%) [locations × 12]
sd_seasonality_data = 'SD_Seasonality.xlsx'            # <- Seasonal std. dev. change (%) [locations × 12]
randomyear_data = 'RandomYearMatrix.xlsx'              # <- (Optional) Pre-generated synthetic year matrix

# === Simulation Configuration ===
numberofyears_syntheticdata = 38   # Number of synthetic years to simulate (one year will be added because of concatenating Z and Z')
startyear_synthetic = 1980         # Start year for synthetic daily data (for leap-year alignment)
alreadyexist = 1                   # 1 = Load existing 'RandomYearMatrix.xlsx'; 0 = Generate new using matrix_year()

# === Scenario Type: Single or Multiple Target Deviations ===
range_flag = 1                     # 0 = single target deviation, 1 = multiple target deviations

# === Deviation Values (Used if range_flag == 1) ===
desired_change_mean = np.arange(-80, 81, 10)  # [min, max+1, step]
desired_change_sd = np.arange(-80, 81, 10)

# === Deviation Values (Used if range_flag == 0) ===
single_deviation_mean = 0              # Single target deviation for monthly means (%)
single_deviation_sd = -10              # Single target deviation for monthly SDs (%)
numberofscenarios_onetarget = 10       # Generate Multiple Scenario for a Single Target Deviation?

# === Boundary Configuration ===         # 1 = Use saved boundary data; 0 = regenerate boundaries for new recorded inflow data
BoundaryCoordinate_AlreadyGenerated = 0  # Set it 0 first to generate and save boundary scenarios. Then set it 1 to use the saved data.

# === Optimization Criteria ===
distance_threshold = 0.01         # Minimum acceptable monthly distance (resultant from target, in %) for early stop in optimization

# === Parallel Optimization ===
enable_parallel = True             # Set to True to enable parallel optimization for locations in each scenario

## === Define Output/Input Paths ===
resultfolder = 'Scenarios'            # Change folder name to save new scenarios in a new folder

## === Saving Timeseries Csv Files ===      # 1 = save; 0 = do not save
daily = 0
monthly = 0

## === Saving .h5 Data of Optimized Scenarios ===      # 1 = save; 0 = do not save 
h5 = 1                                                 # .h5 files are needed for plotting in c1 to c3

# ====================================================================================
#                               End of USER-DEFINED INPUT SECTION
# ====================================================================================
# end

# =================================== Step 1: Load all inputs ===================================
recorded_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', recorded_data))
Data = pd.read_excel(recorded_data_path, sheet_name='Sheet1', header=None).values
Data = Data[1:, :]
Data[Data <= 0] = 0.001

isLocal = pd.read_excel(recorded_data_path, sheet_name='Sheet2', header=None).values
isLocal = isLocal[1:, :]

numberofyears_syntheticdata = numberofyears_syntheticdata + 1  # One extra for appending Z and Z'

# === Load Seasonality Change Factors ===
mean_seasonality_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', mean_seasonality_data))
meanseasonality_change = pd.read_excel(mean_seasonality_path, header=None).values
meanseasonality_change = meanseasonality_change[1:, 1:]
meanseasonality_change = meanseasonality_change.astype(np.float64)

sd_seasonality_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', sd_seasonality_data))
SDseasonality_change = pd.read_excel(sd_seasonality_path, header=None).values
SDseasonality_change = SDseasonality_change[1:, 1:]
SDseasonality_change = SDseasonality_change.astype(np.float64)

# === Exposure Space Bounds Configuration ===
if range_flag == 1:
    p1, p2 = np.meshgrid(desired_change_mean, desired_change_sd)
    desired_scenarios1 = np.column_stack((p1.flatten(), p2.flatten()))
else:
    desired_change = np.zeros((numberofscenarios_onetarget, 2))
    desired_change[:, 0] = single_deviation_mean
    desired_change[:, 1] = single_deviation_sd
    desired_scenarios1 = desired_change

# === Optimization Bounds (Will be used for boundary scenarios generation as well) ===
mean_scenario_range = [-99, 2000]
SD_scenario_range = [-2000, 2000]

range_lb = np.zeros((1, 24))
range_ub = np.zeros((1, 24))
range_lb[0, 0:12] = mean_scenario_range[0]
range_lb[0, 12:24] = SD_scenario_range[0]
range_ub[0, 0:12] = mean_scenario_range[1]
range_ub[0, 12:24] = SD_scenario_range[1]

# === Random Year Matrix Handling ===
randomyear_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', randomyear_data))
if alreadyexist == 1 and os.path.exists(randomyear_path):
    randomyear = pd.read_excel(randomyear_path, header=None).values
else:
    randomyear = matrix_year(Data, numberofyears_syntheticdata)

# === Output/Input Paths ===
scenariofolderpass = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', resultfolder))
os.makedirs(scenariofolderpass, exist_ok=True)
os.makedirs(os.path.join(scenariofolderpass, 'InputData'), exist_ok=True)
os.makedirs(os.path.join(scenariofolderpass, 'OutputData'), exist_ok=True)

# === Boundary Folder ===
boundaryfolderpass = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Boundary'))
os.makedirs(boundaryfolderpass, exist_ok=True)

# === Construct Monthly Exposure Space Matrix ===
desired_scenarios_monthly = np.zeros((desired_scenarios1.shape[0], 24))
for i in range(desired_scenarios1.shape[0]):
    desired_scenarios_monthly[i, 0:12] = desired_scenarios1[i, 0]
    desired_scenarios_monthly[i, 12:24] = desired_scenarios1[i, 1]

# === Process Recorded Data ===
numberoflocations = Data.shape[1] - 3
n = Data.shape[0]
firstyear = int(Data[0, 1])
lastyear = int(Data[-1, 1])
numberofyears_recorded = lastyear - firstyear + 1

# === Initialize Output Variables ===
Monthly_Synthetic = np.zeros((numberofyears_syntheticdata, 12, numberoflocations))
Monthly_Recorded = np.zeros((numberofyears_recorded, 12, numberoflocations))
dist = np.zeros((numberoflocations, 1))
scenario = np.zeros((numberoflocations, 24))
mean_change_syn = np.zeros((numberoflocations, 12))
SD_change_syn = np.zeros((numberoflocations, 12))

# === Save Input Data ===
h5_path = os.path.join(scenariofolderpass, 'InputData', 'Inputs.h5')
with h5py.File(h5_path, 'w') as h5f:
    dset1 = h5f.create_dataset('DailyData[Day x Location]', data=Data[:, 3:].astype(np.float64))
    dset1.attrs['dimension'] = 'Day x Location'
    dset1.attrs['description'] = 'Daily streamflow values for each station'

    dset2 = h5f.create_dataset('NonLocal&Local_Locations[1 x Location]', data=isLocal.astype(np.float64))
    dset2.attrs['dimension'] = '1 x Location'
    dset2.attrs['description'] = '1 = local station, 0 = non-local station'

    dset3 = h5f.create_dataset('Desired_Deviations[Scenario x 2]', data=desired_scenarios1)
    dset3.attrs['dimension'] = 'Scenario x 2'
    dset3.attrs['description'] = 'Each row is a scenario: Column 0 = Mean %, Column 1 = SD %'

    dset4 = h5f.create_dataset('Desired_Sesonality_Mean[Location x Month]', data=meanseasonality_change.astype(np.float64))
    dset4.attrs['dimension'] = 'Location x Month'
    dset4.attrs['description'] = 'Mean seasonality change in % for each location and month'

    dset5 = h5f.create_dataset('Desired_Sesonality_SD[Location x Month]', data=SDseasonality_change.astype(np.float64))
    dset5.attrs['dimension'] = 'Location x Month'
    dset5.attrs['description'] = 'Standard deviation seasonality change in % for each location and month'

    dset6 = h5f.create_dataset('Mean_Forcing_Range[1 x 2]', data=np.array([mean_scenario_range]))
    dset6.attrs['dimension'] = '1 x 2'
    dset6.attrs['description'] = 'Lower and upper bound for mean forcing change (%)'

    dset7 = h5f.create_dataset('SD_Forcing_Range[1 x 2]', data=np.array([SD_scenario_range]))
    dset7.attrs['dimension'] = '1 x 2'
    dset7.attrs['description'] = 'Lower and upper bound for SD forcing change (%)'

    dset8 = h5f.create_dataset('RandomYearMatrix[Year x Month]', data=randomyear.astype(np.float64))
    dset8.attrs['dimension'] = 'Year x Month'
    dset8.attrs['description'] = 'Synthetic year index used for each month of each synthetic year'

    meta_grp = h5f.create_group("Metadata")
    meta_grp.attrs['scenariofolderpass'] = scenariofolderpass.encode("utf-8")
    meta_grp.attrs['boundaryfolderpass'] = boundaryfolderpass.encode("utf-8")
    meta_grp.attrs['numberofyears_syntheticdata+1'] = numberofyears_syntheticdata
    meta_grp.attrs['numberofyears_recorded'] = numberofyears_recorded
    meta_grp.attrs['numberoflocations'] = numberoflocations
    meta_grp.attrs['BoundaryCoordinate_AlreadyGenerated'] = BoundaryCoordinate_AlreadyGenerated
    meta_grp.attrs['range_flag'] = range_flag
    meta_grp.attrs['startyear_synthetic'] = startyear_synthetic

print(f"📁 Input Data Has Been Saved on {resultfolder}\\InputData\\Inputs.h5.")
# =================================== End of Step 1 ===================================

# ================ Step 2: Generating or Loading Boundary Scenarios ================
x_boundary, y_boundary, desired_scenarios_monthly_1 = boundary_coordinate_generator(
    Data, isLocal, numberofyears_syntheticdata, numberofyears_recorded,
    numberoflocations, randomyear, mean_scenario_range, SD_scenario_range,
    boundaryfolderpass, BoundaryCoordinate_AlreadyGenerated
)

# ============== Step 3: Removing Fully Infeasible Scenarios ===============
desired_scenarios_monthly, desired_scenarios1 = remove_infeasible_scenarios(
    scenariofolderpass, desired_scenarios_monthly_1,
    mean_scenario_range, x_boundary, y_boundary,
    desired_scenarios_monthly, desired_scenarios1,
    isLocal, meanseasonality_change, SDseasonality_change, resultfolder
)

# ============ Step 4: Adjusting Partially Infeasible Scenarios =============
adjusted_scenarios = adjust_scenario_to_feasible(
    desired_scenarios_monthly, meanseasonality_change, SDseasonality_change,
    mean_scenario_range, x_boundary, y_boundary,
    numberoflocations, isLocal, desired_scenarios_monthly_1
)

# ============== Step 5: Optimizing Scenarios + Disaggregating To Daily ===============
perform_inverse_optimization_and_disaggregation(
    Data, randomyear, desired_scenarios_monthly, adjusted_scenarios, desired_scenarios1,
    isLocal, meanseasonality_change, SDseasonality_change,
    range_lb, range_ub, range_flag, scenariofolderpass,
    numberofyears_syntheticdata, numberofyears_recorded, numberoflocations,
    firstyear, startyear_synthetic, distance_threshold,
    enable_parallel, resultfolder, daily, monthly, h5
)
