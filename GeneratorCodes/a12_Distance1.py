# a12_Distance1.py

import numpy as np

from a4_Function_Synthetic_Flow_Generator_Monthly import synthetic_flow_generator_monthly
from a5_RecordedMeanSD import recorded_mean_sd
from a6_SyntheticMeanSDChange import synthetic_mean_sd_change

def a12_Distance1(M, S, mon, Data, locationnumber,
                  numberofyears_syntheticdata, randomyear,
                  desired_scenario_monthly, meanseasonality_change1, SDseasonality_change1):
    """
    Objective function used during the optimization of monthly changes.

    This function computes the absolute difference between the simulated and
    adjusted target percent changes in both mean and standard deviation (SD)
    for a specific month and location.

    Parameters:
        M : proposed mean percent change for the month being optimized
        S : proposed SD percent change for the month being optimized
        mon : index of the target month (0 = Jan, 11 = Dec)
        Data : full daily streamflow dataset
        locationnumber : index of the target location
        numberofyears_syntheticdata : number of synthetic years to simulate
        randomyear : matrix assigning each synthetic month to a random year
        desired_scenario_monthly : target monthly mean and sd deviations
        meanseasonality_change1 : seasonal mean target
        SDseasonality_change1 : seasonal SD target

    Returns:
        dist : scalar absolute distance from adjusted target (for the selected month)
        x3 : synthetic monthly time series
        x4 : recorded monthly time series
        mean_change_syn : resultant percent changes in mean 
        SD_change_syn : resultant percent changes in SD 
    """

    # === Step 1: Construct full 24-element scenario vector ===
    scenario1 = np.zeros(24)
    scenario1[mon] = M                 # Mean change for the selected month
    scenario1[mon + 12] = S            # SD change for the selected month

    meanchange = scenario1[:12]        # Extract mean changes
    SDchange = scenario1[12:]          # Extract SD changes

    # === Step 2: Generate synthetic flow data based on this scenario ===
    x3, x4 = synthetic_flow_generator_monthly(
        Data, locationnumber, numberofyears_syntheticdata,
        randomyear, meanchange, SDchange
    )

    # === Step 3: Compute recorded monthly statistics ===
    mean_monthly_recorded, SD_monthly_recorded = recorded_mean_sd(x4)

    # === Step 4: Calculate synthetic changes (relative to recorded) ===
    mean_change_syn, SD_change_syn = synthetic_mean_sd_change(
        x3, mean_monthly_recorded, SD_monthly_recorded
    )

    # === Step 5: Adjust target scenario based on seasonality corrections ===
    # Adjusted = base + linear seasonality + nonlinear seasonality correction
    target_adjusted = desired_scenario_monthly + np.concatenate([
        meanseasonality_change1,
        SDseasonality_change1
    ]) + np.concatenate([
        0.01 * meanseasonality_change1 * desired_scenario_monthly[:12],
        0.01 * SDseasonality_change1 * desired_scenario_monthly[12:]
    ])

    # === Step 6: Compare simulated change with adjusted target ===
    change_monthly = np.concatenate([mean_change_syn, SD_change_syn])
    diff_vector = change_monthly - target_adjusted

    # Distance is the sum of absolute differences for mean and SD of the selected month
    dist = abs(diff_vector[mon]) + abs(diff_vector[mon + 12])

    return dist, x3, x4, mean_change_syn, SD_change_syn