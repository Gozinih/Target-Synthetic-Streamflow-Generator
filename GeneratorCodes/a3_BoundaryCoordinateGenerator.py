# a3_BoundaryCoordinateGenerator.py

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import h5py

from a4_Function_Synthetic_Flow_Generator_Monthly import synthetic_flow_generator_monthly
from a5_RecordedMeanSD import recorded_mean_sd
from a6_SyntheticMeanSDChange import synthetic_mean_sd_change
from a14_ResampleLocals import resample_locals

"""
This function generates a grid of extreme mean/SD deviation scenarios. Then calculates the resulting
changes in synthetic vs. recorded streamflow statistics for each location, as the boundaries for mean and SD.
Saves results as .h5 file in "GeneratorCodes/Boundary" folder for easy use and analysis in python.

Inputs:
    data: Full historical dataset (daily inflow columns by location).
    isLocal: 1D array indicating whether each location is local (1) or non-local (0).
    numberofyears_syntheticdata: Number of synthetic years to generate (includes one extra).
    numberofyears_recorded: Number of historical years available.
    numberoflocations: Total number of inflow locations.
    randomyear: Matrix of randomized year sampling per month [numberofyears_syntheticdata x 12].
    mean_scenario_range: 2-element list [min, max] for mean change in percent.
    SD_scenario_range: 2-element list [min, max] for SD change in percent.
    boundaryfolderpass: Directory path where output should be save ("GeneratorCodes/Boundary") 
    BoundaryCoordinate_AlreadyGenerated: If 1, load saved .h5 boundary data; if 0, regenerate.

Outputs:
    x_boundary: [numberofyears_syntheticdata x 12 x numberoflocations] array of resultant mean changes (%) of the extreme scenarios.
    y_boundary: [numberofyears_syntheticdata x 12 x numberoflocations] array of resultant SD changes (%) of the extreme scenarios.
"""

def boundary_coordinate_generator(data, isLocal, numberofyears_syntheticdata, numberofyears_recorded,
                                  numberoflocations, randomyear, mean_scenario_range, SD_scenario_range,
                                  boundaryfolderpass, BoundaryCoordinate_AlreadyGenerated):

    os.makedirs(boundaryfolderpass, exist_ok=True)

    if BoundaryCoordinate_AlreadyGenerated == 0:
        # === Step 1: Create extreme exposure grid ===
        desired_change_mean_1 = np.arange(-99, mean_scenario_range[1] + 1)
        desired_change_sd_1 = [SD_scenario_range[0]]
        p3, p4 = np.meshgrid(desired_change_mean_1, desired_change_sd_1)
        desired_scenarios1_1 = np.column_stack((p3.flatten(), p4.flatten()))
        num_scenarios = len(desired_scenarios1_1)

        desired_scenarios_monthly_1 = np.zeros((num_scenarios, 24))
        for i in range(num_scenarios):
            desired_scenarios_monthly_1[i, 0:12] = desired_scenarios1_1[i, 0]
            desired_scenarios_monthly_1[i, 12:24] = desired_scenarios1_1[i, 1]
        
        with h5py.File(os.path.join(boundaryfolderpass, 'Boundary_Scenarios.h5'), 'w') as h5f:
            ds1=h5f.create_dataset('Boundary_Scenarios[Scenario x 24]', data=desired_scenarios_monthly_1)
            ds1.attrs['dimension'] = 'Scenario x 24'
            ds1.attrs['description'] = 'Boundary scenarios: forcing Mean (Column 0 to 11) and SD change (Column 12 to 23)'

        # === Initialize outputs ===
        x_boundary = np.zeros((num_scenarios, 12, numberoflocations))
        y_boundary = np.zeros((num_scenarios, 12, numberoflocations))

        for z in tqdm(range(num_scenarios), desc="📊 Generating Boundary Scenarios"):
            meanchange = desired_scenarios_monthly_1[z, :12]
            SDchange = desired_scenarios_monthly_1[z, 12:24]

            mean_change_syn = np.zeros((numberoflocations, 12))
            SD_change_syn = np.zeros((numberoflocations, 12))

            for k in range(numberoflocations):
                if isLocal[0, k] == 0:
                    x3, x4 = synthetic_flow_generator_monthly(
                        data, k, numberofyears_syntheticdata, randomyear, meanchange, SDchange)
                else:
                    x3, x4, _, _, _ = resample_locals(data, k, numberofyears_syntheticdata)

                m_mr, sd_mr = recorded_mean_sd(x4)
                m_cs, sd_cs = synthetic_mean_sd_change(x3, m_mr, sd_mr)

                mean_change_syn[k, :] = m_cs
                SD_change_syn[k, :] = sd_cs

            x_boundary[z, :, :] = mean_change_syn.T
            y_boundary[z, :, :] = SD_change_syn.T

        # === Save results ===
        with h5py.File(os.path.join(boundaryfolderpass, 'Boundary_Coordinates.h5'), 'w') as h5f:
           ds2=h5f.create_dataset('Mean[Scenario x Month x Location]', data=x_boundary)
           ds2.attrs['dimension'] = 'Scenario x Month x Location'
           ds2.attrs['description'] = 'Boundary scenarios: resulted Mean change for different locations and months'
           ds3=h5f.create_dataset('SD[Scenario x Month x Location]', data=y_boundary)
           ds3.attrs['dimension'] = 'Scenario x Month x Location'
           ds3.attrs['description'] = 'Boundary scenarios: resulted SD change for different locations and months'

        print(r"📁 Boundary Coordinates Are Saved on GeneratorCodes\Boundary\Boundary_Coordinates.h5.")

    else:
        print(r"🔄 Loading Previously Generated Boundaries from GeneratorCodes\Boundary\Boundary_Coordinates.h5.")
        with h5py.File(os.path.join(boundaryfolderpass, 'Boundary_Coordinates.h5'), 'r') as h5f:
            x_boundary = h5f['Mean[Scenario x Month x Location]'][:]
            y_boundary = h5f['SD[Scenario x Month x Location]'][:]
        
        with h5py.File(os.path.join(boundaryfolderpass, 'Boundary_Scenarios.h5'), 'r') as h5f:
            desired_scenarios_monthly_1 = h5f['Boundary_Scenarios[Scenario x 24]'][:]

    return x_boundary, y_boundary, desired_scenarios_monthly_1
