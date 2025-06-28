import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import h5py
import sys
from a8_BuildFeasibleAreaPolygon_and_CheckFeasibility import (
    build_all_polygons, check_feasibility_from_polygons
)

def remove_infeasible_scenarios(scenariofolderpass, desired_scenarios_monthly_1,
                                 mean_scenario_range, x_boundary, y_boundary,
                                 desired_scenarios_monthly, desired_scenarios1,
                                 isLocal, meanseasonality_change, SDseasonality_change, resultfolder):
    """
    Remove scenarios that are entirely infeasible across all non-local locations and all months.

    Inputs:
        boundaryfolderpass: Folder containing boundary data
        scenariofolderpass: Folder to save filtered scenarios
        mean_scenario_range: [min, max] range for mean change
        x_boundary, y_boundary: 3D arrays of feasible region bounds
        desired_scenarios_monthly: monthly mean/sd target deviations
        desired_scenarios1: monthly mean/sd target deviations
        isLocal: 1D array indicating local/non-local station flags
        meanseasonality_change: monthly mean target seasonality
        SDseasonality_change: monthly sd target seasonality

    Output:
        desired_scenarios_monthly, desired_scenarios1: Reurn and saves updated scenario arrays (with infeasible ones removed) to .h5.
    """
    change = np.zeros((2,), dtype=np.float64)
    rows_to_delete = []

    nonlocal_indices = [i for i, flag in enumerate(isLocal[0]) if flag == 0]

    # Step 1: Precompute polygon geometry for all (location, month) pairs
    polygon_cache = build_all_polygons(mean_scenario_range, x_boundary, y_boundary,desired_scenarios_monthly_1)

    # Step 2: Loop through scenarios
    for sce in tqdm(range(len(desired_scenarios1)), desc="🔍 Checking Full Feasibility"):
        fully_infeasible = True

        for i, k in enumerate(nonlocal_indices):
            for j in range(12):
                # Apply seasonal deviation adjustments
                change[0] = (desired_scenarios_monthly[sce, j] +
                             meanseasonality_change[i, j] +
                             0.01 * desired_scenarios_monthly[sce, j] * meanseasonality_change[i, j])
                change[1] = (desired_scenarios_monthly[sce, j + 12] +
                             SDseasonality_change[i, j] +
                             0.01 * desired_scenarios_monthly[sce, j + 12] * SDseasonality_change[i, j])

                # Check feasibility using cached polygon
                outside = check_feasibility_from_polygons(change, k, j, polygon_cache)
                if not outside:
                    fully_infeasible = False
                    break
            if not fully_infeasible:
                break

        if fully_infeasible:
            rows_to_delete.append(sce)

    # Remove infeasible scenarios
    desired_scenarios_monthly = np.delete(desired_scenarios_monthly, rows_to_delete, axis=0)
    desired_scenarios1 = np.delete(desired_scenarios1, rows_to_delete, axis=0)

    if desired_scenarios1.size == 0:
        print("❌ All Scenarios Were Infeasible. No Scenarios Remain.")
        sys.exit()

    # Save results
    output_folder = os.path.join(scenariofolderpass, 'OutputData')
    os.makedirs(output_folder, exist_ok=True)

    with h5py.File(os.path.join(output_folder, 'Desired_Scenarios_InfeasibleRemoved.h5'), 'w') as h5f:
       ds1=h5f.create_dataset('Desired_Deviations[Scenario x 2]', data=desired_scenarios1)
       ds1.attrs['dimension'] = 'Scenario x 2'
       ds1.attrs['description'] = 'Each row is a scenario: Column 0 = Mean %, Column 1 = SD %'
       ds1.attrs['columns'] = np.array(['Mean', 'SD'], dtype='S')
       ds2=h5f.create_dataset('Desired_Deviations_Monthly[Scenario x 24]', data=desired_scenarios_monthly)
       ds2.attrs['dimension'] = 'Scenario x 24'
       ds2.attrs['description'] = 'Each row is a scenario: Column 0 to 11 = Mean %, Column 12 to 23 = SD %'
       ds3=h5f.create_dataset('Desired_Sesonality_Mean[Location x Month]', data=meanseasonality_change.astype(np.float64))
       ds3.attrs['dimension'] = 'Location x Month'
       ds3.attrs['description'] = 'Mean seasonality change in % for each location and month'
       ds4=h5f.create_dataset('Desired_Sesonality_SD[Location x Month]', data=SDseasonality_change.astype(np.float64))
       ds4.attrs['dimension'] = 'Location x Month'
       ds4.attrs['description'] = 'SD seasonality change in % for each location and month'

    print(f"✅ From {len(desired_scenarios1)} Scenarios, {len(rows_to_delete)} Fully Infeasible Scenarios Were Removed.")
    print(rf"📁 Desired Scenarios Are Saved on {resultfolder} OutputData\Desired_Scenarios_InfeasibleRemoved.h5.")

    return desired_scenarios_monthly, desired_scenarios1