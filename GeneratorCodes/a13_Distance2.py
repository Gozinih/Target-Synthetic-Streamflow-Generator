# a13_Distance2.py

import numpy as np
from a4_Function_Synthetic_Flow_Generator_Monthly import synthetic_flow_generator_monthly
from a5_RecordedMeanSD import recorded_mean_sd
from a6_SyntheticMeanSDChange import synthetic_mean_sd_change

def a13_Distance2(M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12,
                  S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12,
                  Data, locationnumber, numberofyears_syntheticdata,
                  randomyear, desired_scenario_monthly,
                  meanseasonality_change1, SDseasonality_change1):
    """
    Objective function used after completing all 12 months' optimization.

    Computes the total absolute deviation between the synthetic and target
    changes in both mean and standard deviation across all months.

    Parameters:
        M1 to M12 : optimized monthly percent changes in mean flow (%)
        S1 to S12 : optimized monthly percent changes in standard deviation (%)
        Data : full daily historical streamflow dataset
        locationnumber : index of the target station
        numberofyears_syntheticdata : number of synthetic years to generate
        randomyear : matrix of sampled historical years
        desired_scenario_monthly : target monthly mean and sd deviations
        meanseasonality_change1 : seasonal mean target
        SDseasonality_change1 : seasonal sd target

    Returns:
        dist : scalar total distance (sum of absolute errors)
        x3 : synthetic monthly matrix
        x4 : recorded monthly matrix
        mean_change_syn : resultant percent changes in monthly means
        SD_change_syn : resultant percent changes in monthly SDs
    """

    # === Step 1: Combine inputs into full 24-element scenario vector ===
    scenario1 = np.array([
        M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12,
        S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12
    ])

    meanchange = scenario1[:12]   # Monthly mean change proposals
    SDchange = scenario1[12:]     # Monthly SD change proposals

    # === Step 2: Generate synthetic monthly streamflow ===
    x3, x4 = synthetic_flow_generator_monthly(
        Data, locationnumber, numberofyears_syntheticdata,
        randomyear, meanchange, SDchange
    )

    # === Step 3: Calculate changes between synthetic and recorded stats ===
    mean_monthly_recorded, SD_monthly_recorded = recorded_mean_sd(x4)
    mean_change_syn, SD_change_syn = synthetic_mean_sd_change(
        x3, mean_monthly_recorded, SD_monthly_recorded
    )

    # === Step 4: Compute distance from adjusted target scenario ===
    change_monthly = np.concatenate([mean_change_syn, SD_change_syn])  # Combined result [24]

    # Adjust target by adding linear + nonlinear seasonality modifiers
    target_adjusted = desired_scenario_monthly + np.concatenate([
        meanseasonality_change1,
        SDseasonality_change1
    ]) + np.concatenate([
        0.01 * meanseasonality_change1 * desired_scenario_monthly[:12],
        0.01 * SDseasonality_change1 * desired_scenario_monthly[12:]
    ])

    # === Step 5: Compute total absolute error ===
    dist_vector = change_monthly - target_adjusted
    dist = np.sum(np.abs(dist_vector))

    return dist, x3, x4, mean_change_syn, SD_change_syn