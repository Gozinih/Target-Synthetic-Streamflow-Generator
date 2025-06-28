# a16_SyntheticMonthlytoDailyLocals.py

import numpy as np
from a17_Disaggregation import (
    build_proportion_matrix, disaggregate_monthly_flows, adjust_february
)

def synthetic_monthly_to_daily_locals(Data, Monthly_Synthetic, Monthly_Recorded,
                                      selected_years, firstyear, startyear_synthetic,
                                      local_indices):
    """
    Converts synthetic monthly streamflows for local stations into daily flows
    using historical daily-to-monthly proportions (resampled from recorded data).

    For local stations, disaggregation is based on actual proportions from
    previously selected historical years.

    Parameters:
        Data : full historical daily dataset
        Monthly_Synthetic : synthetic monthly flow matrix
        Monthly_Recorded : recorded monthly flow matrix
        selected_years : array of selected historical years
        firstyear : first year in the historical dataset
        startyear_synthetic : first calendar year for synthetic time series
        numberofyears_syntheticdata : number of synthetic years (includes one extra)
        local_indices : list of indices for local stations

    Returns:
        synthetic_daily : synthetic daily streamflow matrix
    """
    # === Special case: no local stations ===
    if len(local_indices) == 0:
        return np.empty((0, 0))

    # === Step 1: Extract proportions for local station indices ===
    Data_1 = np.asarray(Data[:, 1:], dtype=np.float64)  # Remove day-of-year column
    proportions, years, months = build_proportion_matrix(
        Data_1, Monthly_Recorded, firstyear, local_indices
    )

    # === Step 2: Disaggregate monthly to daily using matched years ===
    synthetic_daily = disaggregate_monthly_flows(
        Monthly_Synthetic, years, months, proportions,
        selected_years.flatten()
    )

    # === Step 3: Adjust February for leap years ===
    synthetic_daily = adjust_february(synthetic_daily, startyear_synthetic)

    return synthetic_daily