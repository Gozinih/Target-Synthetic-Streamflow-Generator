# a15_SyntheticMonthlytoDailynonLocals.py

import numpy as np
from a17_Disaggregation import (
    build_proportion_matrix, disaggregate_monthly_flows, adjust_february
)

def synthetic_monthly_to_daily_nonlocals(Data, Monthly_Synthetic, Monthly_Recorded,
                               firstyear, startyear_synthetic, numberofyears_recorded,
                               nonlocal_indices):
    """
    Converts synthetic monthly flows for non-local stations into synthetic daily flows
    using K-nearest neighbor (KNN) matching and proportional disaggregation.

    This function matches each synthetic year to a historically similar recorded year,
    based on monthly flow similarity, and uses the daily/monthly ratios from that
    historical year to disaggregate monthly flows into daily values.

    Parameters:
        Data : full historical daily dataset
        Monthly_Synthetic : synthetic monthly data
        Monthly_Recorded : recorded monthly data
        firstyear : first year in the recorded dataset
        startyear_synthetic : start year label for synthetic series
        numberofyears_recorded : number of historical years
        nonlocal_indices : list of indices for non-local stations

    Returns:
        synthetic_daily : synthetic daily streamflow
        selected_years : array of selected historical years
    """
    # === Special case: no non-local stations ===
    if len(nonlocal_indices) == 0:
        return np.empty((0, 0)), np.empty((0,), dtype=int)

    # === Step 1: Build proportion matrix (daily/monthly ratios) ===
    Data_1 = np.asarray(Data[:, 1:], dtype=np.float64)
    proportions, years, months = build_proportion_matrix(
        Data_1, Monthly_Recorded, firstyear, nonlocal_indices
    )

    # === Step 2: Select historical years using KNN based on Euclidean distance ===
    n_synth = Monthly_Synthetic.shape[0]          
    n_rec = Monthly_Recorded.shape[0]            

    distances = np.zeros((n_rec, n_synth))        # similarity matrix [recorded x synthetic]

    for i in range(n_rec):
        for j in range(n_synth):
            diff = Monthly_Recorded[i, :, :] - Monthly_Synthetic[j, :, :]
            distances[i, j] = np.sqrt(np.sum(diff ** 2))  # Euclidean distance

    K = int(np.floor(np.sqrt(numberofyears_recorded)))  # number of neighbors
    selected_years = np.zeros(n_synth, dtype=int)       # selected year for each synthetic year

    for j in range(n_synth):
        idx_sorted = np.argsort(distances[:, j])                 # nearest neighbors (by similarity)
        weights = 1.0 / (np.arange(1, K + 1))                    # inverse-rank weights
        probs = weights / np.sum(weights)                        # normalize to probability distribution
        i = np.searchsorted(np.cumsum(probs), np.random.rand())  # stochastic sampling
        selected_years[j] = idx_sorted[i] + firstyear            # convert index to actual year

    # === Step 3: Disaggregate monthly flows to daily using proportions ===
    synthetic_daily = disaggregate_monthly_flows(
        Monthly_Synthetic, years, months, proportions, selected_years
    )

    # === Step 4: Correct February for leap years ===
    synthetic_daily = adjust_february(synthetic_daily, startyear_synthetic)

    return synthetic_daily, selected_years