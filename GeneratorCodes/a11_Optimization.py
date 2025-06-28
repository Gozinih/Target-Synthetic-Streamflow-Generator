# a11_OptimizeForcingScenario.py

import numpy as np
import warnings
from scipy.optimize import differential_evolution

from a12_Distance1 import a12_Distance1
from a13_Distance2 import a13_Distance2
from a14_ResampleLocals import resample_locals

class EarlyStop(Exception):
    pass

def optimize_forcing_scenario(
    Data, k, isLocal_1,
    numberofyears_syntheticdata, randomyear,
    numberofyears_recorded,
    desired_scenario_monthly,
    meanseasonality_change1, SDseasonality_change1,
    range_lb, range_ub,
    Monthly_Synthetic, Monthly_Recorded,
    scenario, dist, mean_change_syn, SD_change_syn, distance_threshold
):
    """
    Optimization of Forcing Scenario for Non-Local Stations

    This script finds the best monthly forcing (mean and SD changes)
    that minimizes the distance to a target synthetic scenario.
    For local stations, only resampling is done (no optimization).

    Parameters:
        Data: Daily streamflow input data
        k: Index of the current location
        sce: Scenario number
        isLocal_1: 1 if the station is local (no optimization), 0 otherwise
        numberofyears_syntheticdata: Number of synthetic years to generate
        randomyear: Random year matrix
        numberofyears_recorded: Number of years in the recorded data
        desired_scenario_monthly: target deviations (12 mean + 12 SD)
        meanseasonality_change1: target mean seasonality
        SDseasonality_change1: target SD seasonality
        range_lb: Lower bounds for optimization
        range_ub: Upper bounds for optimization
        Monthly_Synthetic: 3D array to store synthetic monthly flow
        Monthly_Recorded: 3D array to store recorded monthly flow

    Returns:
        scenario, dist, mean_change_syn, SD_change_syn, Monthly_Synthetic, Monthly_Recorded
    """

    warnings.filterwarnings("ignore", message="delta_grad == 0.0.*")
    x_opt = np.zeros(24)  # Optimized scenario values [12 mean, 12 SD]

    # --- Non-Local Station Optimization ---
    if isLocal_1 == 0:
        for mon in range(12):
            best_distance = [np.inf]
            best_x = [0.0, 0.0]

            # === Define objective function for the optimizer ===
            def objective(x):
                distance = a12_Distance1(
                    x[0], x[1], mon, Data, k,
                    numberofyears_syntheticdata, randomyear,
                    desired_scenario_monthly,
                    meanseasonality_change1, SDseasonality_change1
                )[0]

                if distance < best_distance[0]:
                    best_distance[0] = distance
                    best_x[0], best_x[1] = x[0], x[1]

                if distance < distance_threshold:
                    best_x[0], best_x[1] = x[0], x[1]
                    raise EarlyStop
                
                return distance

            # === Starting guess = desired ===
            initial_guess = [
                desired_scenario_monthly[mon] + meanseasonality_change1[mon]+ 0.01 * desired_scenario_monthly[mon] * meanseasonality_change1[mon],
                desired_scenario_monthly[mon + 12] + SDseasonality_change1[mon] + 0.01 * desired_scenario_monthly[mon + 12] * SDseasonality_change1[mon]
            ]

            # === Define bounds for each variable (mean and SD) ===
            bounds = [
                (range_lb[0, mon], range_ub[0, mon]),
                (range_lb[0, mon + 12], range_ub[0, mon + 12])
            ]
            # === Run optimizer ===
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    strategy='best1bin',
                    maxiter=1000,
                    popsize=15,
                    tol=1e-7,
                    mutation=(0.5, 1),
                    recombination=0.7,
                    polish=True,
                )
                # Store result from optimizer
                x_opt[mon] = result.x[0]
                x_opt[mon + 12] = result.x[1]
            except EarlyStop:
                   x_opt[mon] = best_x[0] 
                   x_opt[mon + 12] = best_x[1]

        # === Evaluate final 24-element scenario ===
        dist1, x3, x4, mean_change_syn1, SD_change_syn1 = a13_Distance2(
            *x_opt,
            Data, k, numberofyears_syntheticdata, randomyear,
            desired_scenario_monthly, meanseasonality_change1, SDseasonality_change1
        )

    # --- Local Station: Resample only (no optimization) ---
    else:
        x3, x4, dist1, mean_change_syn1, SD_change_syn1 = resample_locals(
            Data, k, numberofyears_syntheticdata
        )

    # --- Save Optimized or Resampled Results ---

    # Store optimized forcing vector (12 mean + 12 SD)
    scenario[k, :] = x_opt

    # Store distance from desired, and percent changes in mean/SD
    dist[k] = dist1
    mean_change_syn[k, :] = mean_change_syn1
    SD_change_syn[k, :] = SD_change_syn1

    # Save synthetic monthly flow [years × 12] for this location
    for i in range(numberofyears_syntheticdata - 1):
        for j in range(12):
            Monthly_Synthetic[i, j, k] = x3[i, j]

    # Save recorded monthly flow [years × 12] for this location
    for i in range(numberofyears_recorded):
        for j in range(12):
            Monthly_Recorded[i, j, k] = x4[i, j]

    return scenario, dist, mean_change_syn, SD_change_syn, Monthly_Synthetic, Monthly_Recorded