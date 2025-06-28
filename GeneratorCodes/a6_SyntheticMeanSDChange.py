# a6_SyntheticMeanSDChange.py

import numpy as np

def synthetic_mean_sd_change(x3, mean_monthly_recorded, SD_monthly_recorded):
    """
    Calculates monthly mean and standard deviation changes between synthetic and recorded streamflow data.

    Args:
        x3: Synthetic monthly data matrix for one location
        mean_monthly_recorded, SD_monthly_recorded: Monthly mean and sd of recorded data for the same location

    Returns:
        mean_change_syn, SD_change_syn: monthly meand and sd % change
    """
    mean_monthly_synthetic = np.mean(x3, axis=0)
    SD_monthly_synthetic = np.std(x3, axis=0, ddof=1)

    mean_change_syn = ((mean_monthly_synthetic - mean_monthly_recorded) / mean_monthly_recorded) * 100
    SD_change_syn = ((SD_monthly_synthetic - SD_monthly_recorded) / SD_monthly_recorded) * 100

    return mean_change_syn, SD_change_syn