# a10_InverseApproach_and_MonthlytoDaily.py

import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from joblib import Parallel, delayed

from a2_MatrixYear import matrix_year
from a11_Optimization import optimize_forcing_scenario
from a15_SyntheticMonthlytoDailyNonLocals import synthetic_monthly_to_daily_nonlocals
from a16_SyntheticMonthlytoDailyLocals import synthetic_monthly_to_daily_locals

def perform_inverse_optimization_and_disaggregation(
    Data: np.ndarray,
    randomyear: np.ndarray,
    desired_scenarios_monthly: np.ndarray,
    adjusted_scenarios: np.ndarray,
    desired_scenarios1: np.ndarray,
    isLocal: np.ndarray,
    meanseasonality_change: np.ndarray,
    SDseasonality_change: np.ndarray,
    range_lb: np.ndarray,
    range_ub: np.ndarray,
    range_flag: int,
    scenariofolderpass: str,
    numberofyears_syntheticdata: int,
    numberofyears_recorded: int,
    numberoflocations: int,
    firstyear: int,
    startyear_synthetic: int,
    distance_threshold: float,
    enable_parallel: bool,
    resultfolder: str,
    daily: int,
    monthly: int,
    h5: int
):
    """
    Performs inverse optimization and monthly-to-daily disaggregation for synthetic scenarios.

    Parameters
    ----------
    Data : Historical streamflow data of shape
    randomyear : Random year matrix for synthetic generation
    desired_scenarios_monthly : Target user-defined 24-element deviation scenarios
    adjusted_scenarios : Target adjusted (feasibility adjusted) deviation scenarios
    desired_scenarios1 : Mean/SD deviation summary
    isLocal : 1D array marking local (=1) vs non-local (=0) stations
    meanseasonality_change : monthly mean target seasonality
    SDseasonality_change : monthly sd target seasonality
    range_lb : Lower bounds for optimization, for forcing mean and sd change in a4_
    range_ub : Upper bounds for optimization 
    range_flag : If 0, regenerate `randomyear`; otherwise use existing.
    scenariofolderpass : Output folder path
    numberofyears_syntheticdata : Number of synthetic years
    numberofyears_recorded : Number of historical years in `Data`
    numberoflocations : Number of stations
    firstyear : First calendar year in recorded data
    startyear_synthetic : Starting year label for synthetic data, will be used for adjusting the leap years
    distance_threshold : Optimization early-stop threshold (Euclidean distance)
    enable_parallel : Whether to run location-wise optimizations in parallel
    daily : Save daily inflow time series CSV (1 = yes, 0 = no)
    monthly : Save monthly inflow time series CSV (1 = yes, 0 = no)
    h5 : Save HDF5 output files (1 = yes, 0 = no)
    """
    isLocal = isLocal.flatten()
    local_indices = np.where(isLocal == 1)[0].tolist()
    nonlocal_indices = np.where(isLocal == 0)[0].tolist()
    numlocals = len(local_indices)
    numnonlocals = len(nonlocal_indices)
    n_scenarios = desired_scenarios1.shape[0]

    progress_bar = tqdm(total=n_scenarios, position=0)

    for sce in range(n_scenarios):
        # === Initialize output containers ===
        scenario = np.zeros((numberoflocations, 24))
        dist = np.zeros(numberoflocations)
        mean_change_syn = np.zeros((numberoflocations, 12))
        SD_change_syn = np.zeros((numberoflocations, 12))
        Monthly_Synthetic = np.zeros((numberofyears_syntheticdata - 1, 12, numberoflocations))
        Monthly_Recorded = np.zeros((numberofyears_recorded, 12, numberoflocations))

        # === Generate new random year matrix if required ===
        if range_flag == 0:
            randomyear = matrix_year(Data, numberofyears_syntheticdata)

        # === Optimize each location's forcing scenario ===
        def optimize_one(k):
            monthly_scenario = (
                adjusted_scenarios[sce, :, k] if isLocal[k] == 0
                else desired_scenarios_monthly[sce, :]
            )
            return optimize_forcing_scenario(
                Data, k, isLocal[k],
                numberofyears_syntheticdata, randomyear,
                numberofyears_recorded, monthly_scenario,
                meanseasonality_change[k, :], SDseasonality_change[k, :],
                range_lb, range_ub,
                np.zeros_like(Monthly_Synthetic),
                np.zeros_like(Monthly_Recorded),
                np.zeros_like(scenario),
                np.zeros_like(dist),
                np.zeros_like(mean_change_syn),
                np.zeros_like(SD_change_syn),
                distance_threshold
            )

        if enable_parallel:
            progress_bar.set_description(f"🚀 Parallel Optimization: Scenario {sce+1}/{n_scenarios}")
            results = Parallel(n_jobs=-1)(delayed(optimize_one)(k) for k in range(numberoflocations))
            for k, (scen_k, dist_k, mean_k, sd_k, synth_k, rec_k) in enumerate(results):
                scenario[k, :] = scen_k[k, :]
                dist[k] = dist_k[k]
                mean_change_syn[k, :] = mean_k[k, :]
                SD_change_syn[k, :] = sd_k[k, :]
                Monthly_Synthetic[:, :, k] = synth_k[:, :, k]
                Monthly_Recorded[:, :, k] = rec_k[:, :, k]
        else:
            for k in range(numberoflocations):
                progress_bar.set_description(f"🚀 Optimizing Scenario {sce+1} Loc {k+1}/{numberoflocations}")
                monthly_scenario = (
                    adjusted_scenarios[sce, :, k] if isLocal[k] == 0
                    else desired_scenarios_monthly[sce, :]
                )
                scenario, dist, mean_change_syn, SD_change_syn, Monthly_Synthetic, Monthly_Recorded = optimize_forcing_scenario(
                    Data, k, isLocal[k],
                    numberofyears_syntheticdata, randomyear,
                    numberofyears_recorded, monthly_scenario,
                    meanseasonality_change[k, :], SDseasonality_change[k, :],
                    range_lb, range_ub,
                    Monthly_Synthetic, Monthly_Recorded,
                    scenario, dist, mean_change_syn, SD_change_syn,
                    distance_threshold
                )

        progress_bar.update(1)

        # === Monthly to Daily disaggregation ===
        if numnonlocals > 0:
            section_nonlocal, selected_years = synthetic_monthly_to_daily_nonlocals(
                Data, Monthly_Synthetic[:, :, nonlocal_indices], Monthly_Recorded[:, :, nonlocal_indices],
                firstyear, startyear_synthetic, numberofyears_recorded,
                nonlocal_indices
            )
        else:
            section_nonlocal = np.empty((0, 0))
            selected_years = np.empty((numberofyears_syntheticdata - 1,), dtype=int)

        if numlocals > 0:
            section_local = synthetic_monthly_to_daily_locals(
                Data, Monthly_Synthetic[:, :, local_indices], Monthly_Recorded[:, :, local_indices],
                selected_years.reshape(-1, 1), firstyear, startyear_synthetic,
                local_indices
            )
        else:
            section_local = np.empty((0, 0))

        if section_nonlocal.shape[0] == 0 and section_local.shape[0] == 0:
            raise ValueError("❌ No output generated: both local and non-local sections are empty.")

        base = section_nonlocal if section_nonlocal.shape[0] > 0 else section_local
        year_month = base[:, :2]
        station_cols = [None] * numberoflocations

        if section_nonlocal.shape[1] > 2:
            for idx, k in enumerate(nonlocal_indices):
                station_cols[k] = section_nonlocal[:, 2 + idx].reshape(-1, 1)
        if section_local.shape[1] > 2:
            for idx, k in enumerate(local_indices):
                station_cols[k] = section_local[:, 2 + idx].reshape(-1, 1)

        DailyTimeSeries_Synthetic = np.hstack([year_month] + [station_cols[i] for i in range(numberoflocations)])

        # === Save Outputs ===
        csv_dir = os.path.join(scenariofolderpass, 'DailyTimeseriesCSVFiles')
        monthly_dir = os.path.join(scenariofolderpass, 'MonthlyTimeseriesCSVFiles')
        h5_dir = os.path.join(scenariofolderpass, 'OutputData')
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(monthly_dir, exist_ok=True)
        os.makedirs(h5_dir, exist_ok=True)

        if daily == 1:
            inflow = DailyTimeSeries_Synthetic.copy()
            inflow[:, 2:] = (inflow[:, 2:])
            pd.DataFrame(inflow).to_csv(os.path.join(csv_dir, f'SynDailyInflow_Scenario_{sce+1}.csv'), index=False, header=False)
        
        Monthly_Synthetic = Monthly_Synthetic * 0.0864 # Convert cms.day to MCM
        Monthly_Recorded = Monthly_Recorded * 0.0864

        if monthly == 1:
            years = np.arange(1, numberofyears_syntheticdata)
            ym_grid = np.array([[y, m] for y in years for m in range(1, 13)])
            monthly_flat = Monthly_Synthetic.reshape(-1, numberoflocations)
            monthly_csv = np.hstack([ym_grid, monthly_flat])
            pd.DataFrame(monthly_csv).to_csv(os.path.join(monthly_dir, f'SynMonthlyInflow_Scenario_{sce+1}.csv'), index=False, header=False)

        if h5 == 1:
            with h5py.File(os.path.join(h5_dir, f'Scenario{sce+1}.h5'), 'w') as h5f:
                ds1 = h5f.create_dataset('Opt_Forcing_Scenario[Location x 24]', data=scenario)
                ds1.attrs['dimension'] = 'Location x 24 (12 mean + 12 SD)'
                ds1.attrs['description'] = 'Optimized forcing scenarios (mean and SD) for each location'

                ds2 = h5f.create_dataset('Sum_Distance_FromTarget[Location x 1]', data=dist)
                ds2.attrs['dimension'] = 'Location'
                ds2.attrs['description'] = 'Total Euclidean distance from target per location'

                ds3 = h5f.create_dataset('Monthly_Synthetic(million m3 per month)[Year x Month x Location]', data=Monthly_Synthetic)
                ds3.attrs['dimension'] = 'Year x Month x Location'
                ds3.attrs['description'] = 'Synthetic monthly streamflow for each location and year'

                ds4 = h5f.create_dataset('Monthly_Recorded(million m3 per month)[Year x Month x Location]', data=Monthly_Recorded)
                ds4.attrs['dimension'] = 'Year x Month x Location'
                ds4.attrs['description'] = 'Recorded (historical) monthly streamflow data'

                ds5 = h5f.create_dataset('DailyTimeSeries_Synthetic(m3 per s)[Day x Location]', data=DailyTimeSeries_Synthetic)
                ds5.attrs['dimension'] = 'Day x Location'
                ds5.attrs['description'] = 'Final synthetic daily streamflow per location'

                ds6 = h5f.create_dataset('Opt_Mean_Deviation[Location x Month]', data=mean_change_syn)
                ds6.attrs['dimension'] = 'Location x Month'
                ds6.attrs['description'] = 'Actual mean deviation achieved by optimization for each location/month'

                ds7 = h5f.create_dataset('Opt_SD_Deviation[Location x Month]', data=SD_change_syn)
                ds7.attrs['dimension'] = 'Location x Month'
                ds7.attrs['description'] = 'Actual standard deviation deviation achieved by optimization for each location/month'

                ds8 = h5f.create_dataset('Opt_Sesonality_Mean[Location x Month]', data=meanseasonality_change.astype(np.float64))
                ds8.attrs['dimension'] = 'Location x Month'
                ds8.attrs['description'] = 'Target mean seasonality pattern used in optimization'

                ds9 = h5f.create_dataset('Opt_Sesonality_SD[Location x Month]', data=SDseasonality_change.astype(np.float64))
                ds9.attrs['dimension'] = 'Location x Month'
                ds9.attrs['description'] = 'Target standard deviation seasonality pattern used in optimization'

    progress_bar.close()

    print("✅ All Scenarios Are Generated.")
    if daily == 1: print(rf"📁 Synthetic Daily (m3/s) Timeseries Csv Files Are Saved on {resultfolder} \DailyTimeseriesCSVFiles Folder.")
    if monthly == 1: print(rf"📁 Synthetic Monthly (million m3/month) Timeseries Csv Files Are Saved on {resultfolder} \MonthlyTimeseriesCSVFiles Folder.")
    if h5 == 1:print(rf"📁 Detailed .h5 Data Files Are Saved on {resultfolder} \OutputData Folder.")
